import torch
import torch.nn as nn

class TrainingJob:
    def __init__(self, module: nn.Module, name: str, dataLoader):
        self.module = module
        self.name = name
        self.dataLoader = dataLoader

# A pool runtime that reside perpetually for each GPU in the cluster.
# Launched by ClusterCoordinator
class Runtime:
    def __init__(self, coordinatorAddr, myAddr, device):
        self.coordinatorAddr = coordinatorAddr
        self.myAddr = myAddr
        self.device = device
        self.jobs = []
    
    def scheduleTraining(self, job: TrainingJob):
        self.jobs.append(job)
    
    def run(self):
        print("Starting runtime at %s for device: %s" % (self.myAddr, self.device))
        
        # Event-loop
        # 1. process RPC request
        # 2. run a (or partial) iteration. 

        # Thread per job?
    