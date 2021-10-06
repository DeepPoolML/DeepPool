# Copyright (c) 2021 MIT
# 
# Permission to use, copy, modify, and distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR(S) DISCLAIM ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL AUTHORS BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import subprocess
import json
import sys

# PUBLIC_ADDR_FILENAME="aws_ec2_tools/aws-started-publicDnsName.txt"
# PRIVATE_ADDR_FILENAME="aws_ec2_tools/aws-started-privateIps.txt"

# 172.31.112.9	gitlab.fleet.perspectalabs.com gitlab
# 172.31.112.11	g1dev1.fleet.perspectalabs.com g1dev1
# 172.31.112.12	g1dev2.fleet.perspectalabs.com g1dev2
# 172.31.112.13	g1dev3.fleet.perspectalabs.com g1dev3
# 172.31.112.31	g1lmd1.fleet.perspectalabs.com g1lmd1
# 172.31.112.32	g1lmd2.fleet.perspectalabs.com g1lmd2
# 172.31.112.33	g1lmd3.fleet.perspectalabs.com g1lmd3
# 172.31.112.34	g1lmd4.fleet.perspectalabs.com g1lmd4
# 172.31.112.35	g1lmd5.fleet.perspectalabs.com g1lmd5
# 172.31.112.36	g1lmd6.fleet.perspectalabs.com g1lmd6

# ipAddresses = ["172.31.112.32", "172.31.112.33"] # g1lmd2 and g1lmd3
# ipAddresses = ["172.31.112.33"] # g1lmd3
ipAddresses = ["127.0.0.1"] #172.31.112.34"] # g1lmd4
pkeyPath = '~/.ssh/id_ed25519'
userId = "friedj"
workDir = "~/DeepPoolRuntime/"
gpuIdxOffset = 0
# gpuCount = 1
gpuCount = 8
portPrefix = 11250 # prefix + Device# is used for port.
coordinatorPort = 12347

# with open(PUBLIC_ADDR_FILENAME, "r") as f:
#     publicIps = []
#     for line in f:
#         publicIps.extend(line.split())
# with open(PRIVATE_ADDR_FILENAME, "r") as f:
#     privateIps = []
#     for line in f:
#         privateIps.extend(line.split())

def generateConfigFile():
    # 1. Generate JSON configuration file
    config = {}
    config["workDir"] = workDir
    config["serverList"] = []
    for privateIp in ipAddresses:
        deviceList = []
        for deviceIdx in range(gpuCount):
            portNum = portPrefix + deviceIdx
            deviceList.append({"port": portNum, "device": (deviceIdx + gpuIdxOffset)})
        config["serverList"].append({"addr": privateIp, "deviceList": deviceList, "userId": userId, "sshKeyPath": pkeyPath})
    with open('clusterConfig.json', 'w') as outfile:
        json.dump(config, outfile, indent=2, sort_keys=False)
    print("****** Configuration generated for Fastnic cluster: ")
    print(json.dumps(config, indent=2, sort_keys=False))

def uploadCode():
    # 2. Upload code to AWS servers.
    def upSync(host, localPath, remotePath):
        try:
            # subprocess.check_call(['rsync', '--progress', '-e', 'ssh -i %s -o StrictHostKeyChecking=no' % pkeyPath,
            #     '-rh', "--exclude=*__pycache__", "--exclude=results", localPath, "%s@%s:%s" % (userId, host, remotePath)],
            #     stderr=subprocess.STDOUT)
            subprocess.check_call(['rsync', '--progress', '-e', 'ssh -i %s -o StrictHostKeyChecking=no' % pkeyPath,
                '-rh', "--exclude=*__pycache__", "--exclude=csrc/build/_deps", "--exclude=be_training", "--exclude=be_training/pytorch", "--exclude=be_training/build", "--exclude=results", localPath, "%s@%s:%s" % (userId, host, remotePath)],
                stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            output = e.output
            exit(1)

    for host in ipAddresses:
        upSync(host, ".", workDir)
        print("Uploaded code to %s"%host)

def executeCommand(command):
    def rsh(address, command):
        kwargs = dict()
        kwargs['stderr'] = subprocess.STDOUT
        
        sh_command = ['ssh', '-i', pkeyPath, '-o', 'StrictHostKeyChecking=no', '%s@%s' % (userId, address), '%s' % command]
        try:
            subprocess.check_call(sh_command, **kwargs)
        except subprocess.CalledProcessError as e:
            output = e.output

    for host in ipAddresses:
        rsh(host, command)
        print("Sent \"%s\" to %s"%(command, host))

def downloadResults():
    def downloadFile(address, remotePath: str, localPath: str):
        print("  Downloading %s to %s at %s" % (remotePath, localPath, address))
        sh_command = ['scp', '-i', pkeyPath, '%s@%s:%s' % (userId, address, remotePath), localPath]
        subprocess.check_call(sh_command, stderr=subprocess.STDOUT)

    for host in ipAddresses:
        for remotePath in ["~/DeepPoolRuntime/*.data", "~/DeepPoolRuntime/*.gv.pdf", "~/DeepPoolRuntime/gpuProfile.json"]: #, "~/DeepPoolRuntime/logs/*.out"]: 
            #["~/DeepPoolRuntime/*.data.gv.pdf", "~/DeepPoolRuntime/logs/*.out", "~/*.qdrep", "~/DeepPoolRuntime/logs/*.out", "~/net*.qdrep", "~/net*.sqlite", "~/DeepPoolRuntime/logs/*.out"]:
            try:
                downloadFile(host, remotePath, "results/")
            except subprocess.CalledProcessError as e:
                print(e.output)
            print("Downloaded \"%s\" from %s"%(remotePath, host))

def main():
    downloadResults()
    generateConfigFile()
    uploadCode()
    print("*** To start coordinator, execute following commands ***")
    print("ssh -i %s %s@%s" % (pkeyPath, userId, ipAddresses[0]))
    print("cd %s" % workDir)
    print("python3 cluster.py --addrToBind %s:%d --c10dBackend nccl --be_batch_size=0 --cpp" % (ipAddresses[0], coordinatorPort) )

if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        # executeCommand(sys.argv[1])
        gpuCount = int(sys.argv[1])
        main()
        print("GPU count: %d" % gpuCount)
        executeCommand("w")

    else:
        print("Too many arguments.\nUsage: no args ==> regular cluster setup.\n       one arg ==> execute command to all servers.")
