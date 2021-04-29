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
# import examples.vgg as vgg  # TODO: this is used for debugging. Remove this later.

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
        return None

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
        self.ongoingJobs = {} # Dict of contexts of ongoing jobs. Indexed by job name.
        f = open("runtimeResult.data", "w")
        f.close()
    
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

    def export_scheduleTraining(self, jobName: str, trainingJobInJSON: str):
        job = TrainingJob("test", None, None, 0, "")
        job.loadJSON(trainingJobInJSON)
        print("received job")
        
        gpusUsed = job.getGpusUsed()
        moduleDescList = [job.dumpSingleRunnableModule(rank) for rank in range(gpusUsed)]
        tensorTags = self.buildCommTensorTags(moduleDescList)
        tensorTagsInJson = json.dumps(tensorTags)
        jobRankToGlobalRank = list(range(gpusUsed))
        jobRankToGlobalRankInJson = json.dumps(jobRankToGlobalRank)

        # TODO: should pick locations that doesn't have other priority job scheduled.
        if len(self.locations) < gpusUsed:
            return "Not enough servers available. %d gpus available while %d needed" % (len(self.locations), gpusUsed)

        # convert local ranks to global rank & invoke make groups.
        commGroups = {'all': list(range(gpusUsed))} # TODO: replace this hardcoded one with something like self.buildCommTensorTags(moduleDescList).
        self.initCommGroupsAll(jobName, commGroups, jobRankToGlobalRank)

        threadList = []
        def requestScheduleTraining(proxy, name, jobInJson, dataDir, tensorTagsInJson, jobRankToGlobalRankInJson):
            # print(proxy.scheduleTraining(name, jobInJson, dataDir, tensorTagsInJson, jobRankToGlobalRankInJson))
            proxy.scheduleTraining(name, jobInJson, dataDir, tensorTagsInJson, jobRankToGlobalRankInJson)
        for rank in range(gpusUsed):
            location = self.locations[rank]
            moduleDesc = moduleDescList[rank]
            thread = threading.Thread(name='reqScheTrain%d'%rank, target=requestScheduleTraining, args=(location.getProxy(), jobName, moduleDesc, "SYNTHETIC", tensorTagsInJson, jobRankToGlobalRankInJson))
            threadList.append(thread)
        for thread in threadList:
            thread.start()
            time.sleep(1)
        for thread in threadList:
            thread.join()

        self.ongoingJobs[jobName] = {"iterTime": 0, "gpuUsec": 0, "gpusUsed": gpusUsed, "gpusFinished": 0, "globalBatchSize": job.globalBatchSize}

        # for rank in range(gpusUsed):
        #     location = self.locations[rank]
        #     moduleDesc = moduleDescList[rank] # job.dumpSingleRunnableModule(rank)
        #     print(location.getProxy().scheduleTraining(jobName, moduleDesc, "SYNTHETIC", tensorTagsInJson, jobRankToGlobalRankInJson))
        return 'done'

    def export_notifyTrainingFinished(self, runtimeAddress: str, name: str, remainingJobCount: int, fpTime: float, bpTime: float, iterTime: float):
        print("Training for %s is completed at %s. (%d jobs are remaining) fp: %3.1f bp: %3.1f iterTime: %3.1f" % (name, runtimeAddress, remainingJobCount, fpTime, bpTime, iterTime))
        self.ongoingJobs[name]["iterTime"] = max(self.ongoingJobs[name]["iterTime"], iterTime)
        self.ongoingJobs[name]["gpuUsec"] += fpTime + bpTime
        self.ongoingJobs[name]["gpusFinished"] += 1
        if self.ongoingJobs[name]["gpusFinished"] == self.ongoingJobs[name]["gpusUsed"]:
            print("Training for %s is completed entirely. GpusUsed: %d  IterTime: %3.1f ms  GpuMsec: %3.1f ms" %
                    (name, self.ongoingJobs[name]["gpusUsed"], self.ongoingJobs[name]["iterTime"] / 1000, self.ongoingJobs[name]["gpuUsec"] / 1000))
            print("  %2d    %2d    %3.1f   %3.1f " %
                    (self.ongoingJobs[name]["globalBatchSize"], self.ongoingJobs[name]["gpusUsed"], self.ongoingJobs[name]["iterTime"] / 1000, self.ongoingJobs[name]["gpuUsec"] / 1000))
            f = open("runtimeResult.data", "a")
            f.write("  %2d    %2d   %4.1f  %4.1f\n" %
                    (self.ongoingJobs[name]["globalBatchSize"], self.ongoingJobs[name]["gpusUsed"], self.ongoingJobs[name]["iterTime"] / 1000, self.ongoingJobs[name]["gpuUsec"] / 1000))
            f.close()
        return 'done'

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
        
    def launchRuntimeAll(self, c10dBackend: str, profile: bool):
        """ Launch runtime at all remote locations. Also registers the sighandler
            that cleanly shuts down all remote runtime servers.
        """
        for i, location in enumerate(self.locations):
            location.upSync(".", self.workDir)
            # pass master ip and port.
            stdoutFp = open("logs/runtime%d.out"%i, "w", buffering=1)
            stderrFp = open("logs/runtime%d.err"%i, "w", buffering=1)
            if profile:# and location.device == 0: # Only run 1 nsys per host.
                nsysPrefix = "nsys profile -f true -o net%d -c cudaProfilerApi --stop-on-range-end true -t cuda,nvtx --export sqlite " % i # -s none
            else:
                nsysPrefix = ""
            self.processes.append(location.rshAsync(
                nsysPrefix + "python3 " + self.workDir + "runtime.py" + \
                " --coordinatorAddr %s:%d --myAddr %s:%d --device %d --c10dBackend %s --rank %d --worldSize %d" % \
                    (self.myAddr, self.myPort, location.address, location.port, location.device, c10dBackend, i, len(self.locations)) #+ \
                , stdout=stdoutFp, stderr=stderrFp))

            sig_names = {2: "SIGINT", 15: "SIGTERM"}
            last_return_code = None
            def sigkill_handler(signum, frame):
                print("signum:%d Trying to shutdown all runtime." % signum)
                self.shutdownRuntimeAll()
                # self.waitForRuntimeAll()
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
            # signal.signal(signal.SIGTERM, sigkill_handler)
        
        time.sleep(5 + (15 if profile else 0))
        for location in self.locations:
            print(location.getProxy().poke())


    def shutdownRuntimeAll(self):
        """ Ask all remote runtime servers to stop. Returns after all servers ack the shutdown request. """
        for location in self.locations:
            try:
                proxy = location.getProxy(maxRetry=1)
                if proxy != None:
                    print(proxy.shutdown())
                # print(location.getProxy(maxRetry=1).shutdown())
            except xmlrpc.client.Fault:
                print("pipe broken while shuting down %s" % location.address)

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

    def initCommGroupsAll(self, jobName: str, commGrpDict: dict, jobRankToGlobalRank: list):
        """ A helper function that will ask all runtimes to create new c10d comm groups.
            Used while scheduling a new training job. This method should be invoked before
            scheduling a new training job to any runtime that will participate in training.
        """

        commGrpDictWithGlobalRanks = {}
        for grpName in commGrpDict:
            grpRanks = commGrpDict[grpName]
            globalGrpRanks = [jobRankToGlobalRank[rank] for rank in grpRanks]
            commGrpDictWithGlobalRanks[grpName] = globalGrpRanks
        commGrpDictWithGlobalRanksInJson = json.dumps(commGrpDictWithGlobalRanks)

        threadList = []
        def requestInitCommGroups(proxy, jobName, commGroupsInJson):
            # print(proxy.initCommGroups(jobName, commGroupsInJson))
            proxy.initCommGroups(jobName, commGroupsInJson)
        for i, location in enumerate(self.locations):
            thread = threading.Thread(name='init_commGroups%d'%i, target=requestInitCommGroups,
                                      args=(location.getProxy(), jobName, commGrpDictWithGlobalRanksInJson,))
            threadList.append(thread)
        for thread in threadList:
            thread.start()
            time.sleep(1)
        for thread in threadList:
            thread.join()
            

    def waitForRuntimeAll(self):
        """ Waits until all runtime processes terminate. Development use only. """
        # TODO: replace this method with xmlrpc server event loop.
        print("Waiting for ssh process to terminate.")
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
    parser.add_argument("--logLevel", type=int, default=1,
                        help="Logging level. 0: verbose, 1: Info, 2: Error") # NOT YET IMPLEMENTED.
    parser.add_argument("--pathToConfig", type=str, default="clusterConfig.json",
                        help="The full path to the cluster configuration files")
    parser.add_argument('--install', default=False, action='store_true',
                        help="When this option is set, it will install required pip packages to all servers")
    parser.add_argument('--profile', default=False, action='store_true',
                        help="To launch runtimes with night system profiling.")
    # For installing nsys.. (with other cuda toolkit..)
    # wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
    # sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
    # sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    # sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
    # sudo apt-get update
    # sudo apt-get -y install cuda

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
    # Just make sure there's no previously left runtimes.
    print("Cleaning up potentially leftover runtime servers from previous experiment.")
    coordinator.shutdownRuntimeAll()
    time.sleep(10)

    coordinator.launchRuntimeAll(args.c10dBackend, profile=args.profile)
    print("All runtime nodes are up and running. Now, initializing communication backend..")
    time.sleep(5)
    coordinator.initCommBackendAll()
    print("Communication backends are ready at all locations.")
    print("Now, cluster is ready to accept training jobs.")

    # def submitVGG():
    #     job = vgg.genTestJob(1, 16)
    #     coordinator.export_scheduleTraining("vggLocal", job.dumpInJSON())
    # thread = threading.Thread(name='vgg.main', target=submitVGG)
    # thread.start()

    coordinator.serve_forever()

    # thread.join()

if __name__ == "__main__":
    main()