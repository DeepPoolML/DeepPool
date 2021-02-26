import subprocess
import json

PUBLIC_ADDR_FILENAME="../aws-started-publicDnsName.txt"
PRIVATE_ADDR_FILENAME="../aws-started-privateIps.txt"
pkeyPath = '~/.ssh/ulma-sjp.pem'
userId = "ubuntu"
workDir = "~/DeepPoolRuntime/"
gpuCount = 1

with open(PUBLIC_ADDR_FILENAME, "r") as f:
    publicIps = []
    for line in f:
        publicIps.extend(line.split())
with open(PRIVATE_ADDR_FILENAME, "r") as f:
    privateIps = []
    for line in f:
        privateIps.extend(line.split())

# 1. Generate JSON configuration file
config = {}
config["workDir"] = workDir
config["serverList"] = []
for privateIp in privateIps:
    config["serverList"].append({"addr": privateIp, "port": 1234, "gpuCount": gpuCount, "userId": userId, "sshKeyPath": pkeyPath})
with open('clusterConfig.json', 'w') as outfile:
    json.dump(config, outfile, indent=2, sort_keys=False)
print("****** Configuration generated for AWS cluster: ")
print(json.dumps(config, indent=2, sort_keys=False))

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

print("ssh -i %s %s@%s" % (pkeyPath, userId, publicIps[0]))
