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

import time
import signal
import sys
import subprocess
import json
import xmlrpc.server
import xmlrpc.client
import re
import threading
from argparse import ArgumentParser, REMAINDER
from typing import Optional, IO, List, Any
from jobDescription import TrainingJob

class Location:
    def __init__(self, address: str, port: int, device: int, userId: str, sshKeyPath: str):
        self.address = address
        self.port = port
        self.device = device
        self.userId = userId
        self.sshKeyPath = sshKeyPath
        self.serverId = None
        self.proxy = None
        
    def getProxy(self, maxRetry = 8):
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
        sh_command = ['ssh', '-i', self.sshKeyPath, '-o', 'StrictHostKeyChecking=no', '%s@%s' % (self.userId, self.address), '%s' % command]
        try:
            subprocess.check_call(sh_command, **kwargs)
        except subprocess.CalledProcessError as e:
            output = e.output
            exit(1)
        return
    
    def rshAsync(self, command, **kwargs):
        print("Sending cmd: %s" % command)
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

class ClusterClient:
    """ A handle to submit training job to cluster. """

    def __init__(self, coordinatorAddr: str, coordinatorPort: int, maxRetries = 5):
        retryGap = 1
        retryCount = 0
        while retryCount < maxRetries:
            try:
                self.proxy = xmlrpc.client.ServerProxy("http://%s:%d/"%(coordinatorAddr, coordinatorPort))
                self.proxy.poke()
                return
            except ConnectionRefusedError:
                print("Cannot connect to %s:%d. Will retry in %d sec." %
                    (coordinatorAddr, coordinatorPort, retryGap))
                time.sleep(retryGap)
                retryGap *= 2 # exponential back off.
                retryCount += 1
    
    def submitTrainingJob(self, trainingJobInJSON: str):
        self.proxy.poke()
        self.proxy.scheduleTraining(trainingJobInJSON)



class ClusterCoordinator(xmlrpc.server.SimpleXMLRPCServer):
    """ GPU cluster coordinator. It accepts training jobs from clients and schedule them to runtimes. """

    def __init__(self, addrToBind: str, portToBind: int, locations: List[Location], workDir: str):
        super(ClusterCoordinator, self).__init__((addrToBind, portToBind))
        self.myAddr = addrToBind
        self.myPort = portToBind
        self.locations = locations
        self.workDir = workDir
        self.processes = []  # from subprocess calls used for launching runtime.
        self.nextTagStartOffset = 1
    
    def _dispatch(self, method, params):
        """ Custom dispatcher for XML-RPC server. """
        try:
            # We are forcing the 'export_' prefix on methods that are
            # callable through XML-RPC for security.
            func = getattr(self, 'export_' + method)
        except AttributeError:
            raise Exception('method "%s" is not supported' % method)
        else:
            return func(*params)

    ######################################################
    ## RPC handlers
    ######################################################
    def export_poke(self):
        return 'Returned from poke at %s' % self.myAddr

    def export_scheduleTraining(self, trainingJobInJSON: str):
        job = TrainingJob("test", None, None, 0, "")
        job.loadJSON(trainingJobInJSON)
        print("received job")
        
        gpusUsed = job.getGpusUsed()
        moduleDescList = [job.dumpSingleRunnableModule(rank) for rank in range(gpusUsed)]
        tensorTags = self.buildCommTensorTags(moduleDescList)
        tensorTagsInJson = json.dumps(tensorTags)

        # TODO: should pick locations that doesn't have other priority job scheduled.
        if len(self.locations) < gpusUsed:
            return "Not enough servers available. %d gpus available while %d needed" % (len(self.locations), gpusUsed)
        
        for rank in range(gpusUsed):
            location = self.locations[rank]
            moduleDesc = moduleDescList[rank] # job.dumpSingleRunnableModule(rank)
            print(location.getProxy().scheduleTraining("vgg", moduleDesc, "SYNTHETIC", tensorTagsInJson))
        return 'done'

    def export_notifyTrainingFinished(self, runtimeAddress: str, name: str, remainingJobCount: int):
        print("Training for %s is completed at %s. (%d jobs are remaining)" % (name, runtimeAddress, remainingJobCount))
        # return 'done'

    def export_addGpuNode(self):
        print("NOT YET IMPLEMENTED.")

    ######################################################
    ## Internal helper methods
    ######################################################
    def buildCommTensorTags(self, moduleDescList):
        # TODO: need tag allocator that can recycle tags.
        tag = self.nextTagStartOffset
        tensorTags = {}
        for moduleDesc in moduleDescList:
            spec = json.loads(moduleDesc)
            
            for ldsc in spec["layers"]:
                if "tensorRx" in ldsc: # either sender or receiver need to assign tag.
                    for item in ldsc["tensorRx"]:
                        tensorTags[item["name"]] = tag
                        tag += 1
                        tensorTags[item["name"] + "_back"] = tag
                        tag += 1
        self.nextTagStartOffset = (tag + 99) % 100
        return tensorTags

    ######################################################
    ## Runtime cluster management
    ######################################################
    def installPackages(self):
        """ Install required software at each runtime server """
        pipPackages = ["torch", "jsonpickle", "torchvision"]
            # "pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html"]
        for location in self.locations:
            for pipPackage in pipPackages:
                location.rsh("pip install %s" % pipPackage)
        
    def launchRuntimeAll(self, c10dBackend: str):
        """ Launch runtime at all remote locations. Also registers the sighandler
            that cleanly shuts down all remote runtime servers.
        """
        for i, location in enumerate(self.locations):
            location.upSync(".", self.workDir)
            # pass master ip and port.
            stdoutFp = open("logs/runtime%d.out"%i, "w", buffering=1)
            stderrFp = open("logs/runtime%d.err"%i, "w", buffering=1)
            self.processes.append(location.rshAsync(
                "python3 " + self.workDir + "runtime.py" + \
                " --coordinatorAddr %s:%d --myAddr %s:%d --device %d --c10dBackend %s --rank %d --worldSize %d" % \
                    (self.myAddr, self.myPort, location.address, location.port, location.device, c10dBackend, i, len(self.locations)) #+ \
                , stdout=stdoutFp, stderr=stderrFp))

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
        
        for location in self.locations:
            print(location.getProxy().poke())


    def shutdownRuntimeAll(self):
        """ Ask all remote runtime servers to stop. Returns after all servers ack the shutdown request. """
        for location in self.locations:
            print(location.getProxy().shutdown())

    def initCommBackendAll(self):
        threadList = []
        def requestInitCommBackend(proxy):
            print(proxy.initCommBackend())
        for i, location in enumerate(self.locations):
            thread = threading.Thread(name='init_comm%d'%i, target=requestInitCommBackend, args=(location.getProxy(),))
            threadList.append(thread)
        for thread in threadList:
            thread.start()
            time.sleep(1)
        for thread in threadList:
            thread.join()

    def waitForRuntimeAll(self):
        """ Waits until all runtime processes terminate. Development use only. """
        # TODO: replace this method with xmlrpc server event loop.
        for p in self.processes:
            p.wait()


####################################################################################
##  Initial launch scripts
####################################################################################

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
    parser.add_argument("--c10dBackend", type=str, default="nccl",
                    help="pytorch c10d communication backend. Type either nccl or gloo")

    # node_local_rank_stdout_filename = "node_{}_local_rank_{}_stdout"
    # node_local_rank_stderr_filename = "node_{}_local_rank_{}_stderr"
    # parser.add_argument(
    #     "--logdir",
    #     default=None,
    #     type=str,
    #     help=f"""Relative path to write subprocess logs to. Passing in a relative
    #     path will create a directory if needed, and write the stdout and stderr to files
    #     {node_local_rank_stdout_filename} and {node_local_rank_stderr_filename}. Note that
    #     successive runs with the  same path to write logs to will overwrite existing logs,
    #     so be sure to save logs as needed.""",
    # )
    parser.add_argument("--pathToConfig", type=str, default="clusterConfig.json",
                        help="The full path to the cluster configuration files")
    parser.add_argument('--install', default=False, action='store_true',
                        help="When this option is set, it will install required pip packages to all servers")

    return parser.parse_args()

def main():
    args = parse_args()
    clusterConfig = json.load(open(args.pathToConfig))
    locations = []
    for serverConfig in clusterConfig["serverList"]:
        print("Found %s" % str(serverConfig))
        for deviceConfig in serverConfig["deviceList"]:
            locations.append(Location(serverConfig["addr"], deviceConfig["port"], deviceConfig["device"], serverConfig["userId"], serverConfig["sshKeyPath"]))
    addrToBindCombo = re.split('[-:]', args.addrToBind)
    addrToBind = addrToBindCombo[0]
    portToBind = int(addrToBindCombo[1])

    coordinator = ClusterCoordinator(addrToBind, portToBind, locations, clusterConfig["workDir"])
    if args.install:
        coordinator.installPackages()
    coordinator.launchRuntimeAll(args.c10dBackend)
    print("Cluster initialization completed.")
    time.sleep(5)
    coordinator.initCommBackendAll()
    print("Communication backends are ready at all locations.")
    print("Now, cluster is ready to accept training job.")
    coordinator.serve_forever()
    coordinator.shutdownRuntimeAll()
    coordinator.waitForRuntimeAll()

    # TODO: listen port to changes on config?

if __name__ == "__main__":
    main()