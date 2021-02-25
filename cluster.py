


class Location:
    def __init__(self, address, device):
        self.address = address
        self.device = device
    
class ClusterCoordinator:
    def __init__(self, locations: list):
        self.locations = locations
    
    # Launch runtime at
    # Don't use this for now for easier debugging?
    def launchRuntime(self, address, device):
        # Refer torch.distributed.launch

    def scheduleTraining(self): #TODO: Argument for this? CostSim?
        # TODO: how to serialize RunnableModule?

