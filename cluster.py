import time
import signal
import sys
import subprocess
import os
import json
import xmlrpc.server
import xmlrpc.client
import re
from argparse import ArgumentParser, REMAINDER
from typing import Optional, IO, List, Any


class Location:
    def __init__(self, address: str, port: int, device: int, userId: str, sshKeyPath: str):
        self.address = address
        self.port = port
        self.device = device
        self.userId = userId
        self.sshKeyPath = sshKeyPath
        self.serverId = None
        self.proxy = None
        
    def getProxy(self, maxRetry = 5):
        if self.proxy != None:
            return self.proxy
        retryGap = 1
        retryCount = 0
        while retryCount < maxRetry:
            try:
                self.proxy = xmlrpc.client.ServerProxy("http://%s:%d/"%(self.address, self.port))
                self.proxy.poke()
                return self.proxy
            except ConnectionRefusedError:
                print("Cannot connect to %s:%d. Will retry in %d sec." %
                    (self.address, self.port, retryGap))
                time.sleep(retryGap)
                retryGap *= 2 # exponential back off.
                retryCount += 1

    def downloadFile(self, remotePath: str, localPath: str):
        print("  Downloading %s to %s at %s" % (remotePath, localPath, self.address))
        kwargs = dict()
        kwargs['stderr'] = subprocess.STDOUT
        # sh_command = ['mkdir', '-p', localPath]
        # subprocess.check_call(sh_command, **kwargs)
        sh_command = ['scp', '-i', self.sshKeyPath, '%s@%s:%s' % (self.userId, self.address, remotePath), localPath]
        subprocess.check_call(sh_command, **kwargs)

    def uploadFile(self, localFilePath, remotePath):
        print("  Uploading %s to %s at %s" % (localFilePath, remotePath, self.address))
        kwargs = dict()
        # kwargs['shell'] = True
        kwargs['stderr'] = subprocess.STDOUT
        sh_command = ['scp', '-i', self.sshKeyPath, localFilePath, '%s@%s:%s' % (self.userId, self.address, remotePath)]
        subprocess.check_call(sh_command, **kwargs)
    
    def rsh(self, command):
        kwargs = dict()
        kwargs['stderr'] = subprocess.STDOUT
        # sh_command = ['ssh', '-v', '-i', '~/.ssh/ulma-sjp.pem', 'ubuntu@%s' % self, '%s' % command]
        sh_command = ['ssh', '-i', self.sshKeyPath, '%s@%s' % (self.userId, self.address), '%s' % command]
        try:
            subprocess.check_call(sh_command, **kwargs)
        except subprocess.CalledProcessError as e:
            output = e.output
            exit(1)
        return
    
    def rshAsync(self, command, **kwargs):
        sh_command = ['ssh', '-i', self.sshKeyPath, '%s@%s' % (self.userId, self.address),
                    '%s' % command]
        p = subprocess.Popen(sh_command, **kwargs)
        return p

    def upSync(self, localPath, remotePath):
        try:
            subprocess.check_call(['rsync', '-e', 'ssh -i %s -o StrictHostKeyChecking=no' % self.sshKeyPath,
                '-rh', "--exclude=*__pycache__", localPath, "%s@%s:%s" % (self.userId, self.address, remotePath)],
                stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            output = e.output
            exit(1)

class ClusterCoordinator:
    def __init__(self, addrToBind: str, portToBind: int, locations: List[Location], workDir: str):
        self.myAddr = addrToBind
        self.myPort = portToBind
        self.locations = locations
        self.workDir = workDir
        self.processes = []  # from subprocess calls used for launching runtime.

    def installPackages(self):
        pipPackages = ["torch"]
        for location in self.locations:
            for pipPackage in pipPackages:
                location.rsh("pip install %s" % pipPackage)
        
    def launchRuntimeAll(self):
        for location in self.locations:
            location.upSync(".", self.workDir)
            # pass master ip and port.
            self.processes.append(location.rshAsync(
                "python3 " + self.workDir + "runtime.py" + \
                    " --coordinatorAddr %s:%d --myAddr %s:%d --device %d" % \
                        (self.myAddr, self.myPort, location.address, location.port, location.device) ))
            print(location.getProxy().poke())

            sig_names = {2: "SIGINT", 15: "SIGTERM"}
            last_return_code = None
            def sigkill_handler(signum, frame):
                self.shutdownRuntimeAll()
                for process in self.processes:
                    print(f"Killing subprocess {process.pid}")
                    try:
                        process.terminate()
                        # process.kill()
                    except Exception:
                        pass
                if last_return_code is not None:
                    raise subprocess.CalledProcessError(returncode=last_return_code, cmd=cmd)
                if signum in sig_names:
                    print(f"Main process received {sig_names[signum]}, exiting")
                sys.exit(1)
            signal.signal(signal.SIGINT, sigkill_handler)
            signal.signal(signal.SIGTERM, sigkill_handler)

    def shutdownRuntimeAll(self):
        for location in self.locations:
            # print("Shutdown requested to %s:%d" % (location.address, location.port))
            with xmlrpc.client.ServerProxy("http://%s:%d/"%(location.address, location.port)) as proxy:
                print(proxy.shutdown())
                print("Shutdown requested to %s:%d" % (location.address, location.port))

    def waitForRuntimeAll(self):
        # TODO: remove this later when finished coordinator server implementation.
        for p in self.processes:
            p.wait()
    
    def scheduleTraining(self): #TODO: Argument for this? CostSim?
        # TODO: how to serialize RunnableModule?
        print("NOT YET IMPLEMENTED.")


####################################################################################
##  One-time launch scripts
####################################################################################
node_local_rank_stdout_filename = "node_{}_local_rank_{}_stdout"
node_local_rank_stderr_filename = "node_{}_local_rank_{}_stderr"

def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="ClusterCoordinator initial launch "
                                        "script that will spawn up "
                                        "multiple distributed processes")

    # Optional arguments for the launch helper
    parser.add_argument("--addrToBind", type=str, default="localhost:12340",
                        help="IP:port to listen for requests to the cluster coordinator")
    parser.add_argument("--nnodes", type=int, default=1,
                        help="The number of nodes to use for distributed "
                             "training")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="The number of processes to launch on each node, "
                             "for GPU training, this is recommended to be set "
                             "to the number of GPUs in your system so that "
                             "each process can be bound to a single GPU.")
    parser.add_argument(
        "--logdir",
        default=None,
        type=str,
        help=f"""Relative path to write subprocess logs to. Passing in a relative
        path will create a directory if needed, and write the stdout and stderr to files
        {node_local_rank_stdout_filename} and {node_local_rank_stderr_filename}. Note that
        successive runs with the  same path to write logs to will overwrite existing logs,
        so be sure to save logs as needed.""",
    )
    parser.add_argument("--pathToConfig", type=str, default="clusterConfig.json",
                        help="The full path to the cluster configuration files")
    return parser.parse_args()

def main():
    args = parse_args()
    clusterConfig = json.load(open(args.pathToConfig))
    locations = []
    for serverConfig in clusterConfig["serverList"]:
        print("Found %s" % str(serverConfig))
        for deviceIdx in range(serverConfig["gpuCount"]):
            locations.append(Location(serverConfig["addr"], serverConfig["port"], deviceIdx, serverConfig["userId"], serverConfig["sshKeyPath"]))
    addrToBindCombo = re.split('[-:]', args.addrToBind)
    addrToBind = addrToBindCombo[0]
    portToBind = int(addrToBindCombo[1])

    coordinator = ClusterCoordinator(addrToBind, portToBind, locations, clusterConfig["workDir"])
    # coordinator.installPackages()
    coordinator.launchRuntimeAll()
    print("Cluster initialization completed.")
    time.sleep(50)
    coordinator.shutdownRuntimeAll()
    coordinator.waitForRuntimeAll()

    # TODO: listen port to changes on config?

if __name__ == "__main__":
    main()