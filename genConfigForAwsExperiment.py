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

PUBLIC_ADDR_FILENAME="aws_ec2_tools/aws-started-publicDnsName.txt"
PRIVATE_ADDR_FILENAME="aws_ec2_tools/aws-started-privateIps.txt"
pkeyPath = '~/.ssh/ulma-sjp.pem'
userId = "ubuntu"
workDir = "~/DeepPoolRuntime/"
gpuCount = 1
portPrefix = 1140 # prefix + Device# is used for port.
coordinatorPort = 12345

with open(PUBLIC_ADDR_FILENAME, "r") as f:
    publicIps = []
    for line in f:
        publicIps.extend(line.split())
with open(PRIVATE_ADDR_FILENAME, "r") as f:
    privateIps = []
    for line in f:
        privateIps.extend(line.split())

def generateConfigFile():
    # 1. Generate JSON configuration file
    config = {}
    config["workDir"] = workDir
    config["serverList"] = []
    for privateIp in privateIps:
        deviceList = []
        for deviceIdx in range(gpuCount):
            portNum = portPrefix + deviceIdx
            deviceList.append({"port": portNum, "device": deviceIdx})
        config["serverList"].append({"addr": privateIp, "deviceList": deviceList, "userId": userId, "sshKeyPath": pkeyPath})
    with open('clusterConfig.json', 'w') as outfile:
        json.dump(config, outfile, indent=2, sort_keys=False)
    print("****** Configuration generated for AWS cluster: ")
    print(json.dumps(config, indent=2, sort_keys=False))

def uploadCode():
    # 2. Upload code to AWS servers.
    def upSync(host, localPath, remotePath):
        try:
            subprocess.check_call(['rsync', '-e', 'ssh -i %s -o StrictHostKeyChecking=no' % pkeyPath,
                '-rh', "--exclude=*__pycache__", localPath, "%s@%s:%s" % (userId, host, remotePath)],
                stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            output = e.output
            exit(1)

    for host in publicIps:
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

    for host in publicIps:
        rsh(host, command)
        print("Sent \"%s\" to %s"%(command, host))

def main():
    generateConfigFile()
    uploadCode()
    print("*** To start coordinator, execute following commands ***")
    print("ssh -i %s %s@%s" % (pkeyPath, userId, publicIps[0]))
    print("cd %s" % workDir, end=" && ")
    print("python3 cluster.py --addrToBind %s:%d" % (privateIps[0], coordinatorPort) )

if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        executeCommand(sys.argv[1])
    else:
        print("Too many arguments.\nUsage: no args ==> regular cluster setup.\n       one arg ==> execute command to all servers.")
