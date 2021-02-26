import xmlrpc.server
import torch
import torch.nn as nn
import re
import threading
import signal
import sys
import time
from argparse import ArgumentParser, REMAINDER

class TrainingJob:
    def __init__(self, module: nn.Module, name: str, dataLoader):
        self.module = module
        self.name = name
        self.dataLoader = dataLoader

class Runtime(xmlrpc.server.SimpleXMLRPCServer):
    """A pool runtime that reside perpetually for each GPU in the cluster.
    
    This class is launched by ClusterCoordinator.
    """

    # class RequestHandler(xmlrpc.server.SimpleXMLRPCRequestHandler):
    #     rpc_paths = ('/Coordinator2Runtime',)

    def __init__(self, coordinatorAddr: str, coordinatorPort: int, myAddr: str, myPort: int, device: int):
        super(Runtime, self).__init__((myAddr, myPort))#, requestHandler=Runtime.RequestHandler)
        self.coordinatorAddr = coordinatorAddr
        self.coordinatorPort = coordinatorPort
        self.myAddr = myAddr
        self.myPort = myPort
        self.device = device
        self.jobs = []
        self.pollInvokeCounter = 0
        print("Runtime initialized with coordAddr=%s:%d, myAddr=%s:%d, device=%d" %
            (coordinatorAddr, coordinatorPort, myAddr, myPort, device) )
    
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
    def export_scheduleTraining(self, job: TrainingJob):
        self.jobs.append(job)
        print("Invoked scheduleTraining @ %s!"%self.myAddr)

    def export_poke(self):
        print("poked! at %s."%self.myAddr)
        return 'Returned from poke at %s' % self.myAddr

    def export_shutdown(self):
        self.shutdownRequested = True
        return 'Returned from remote_shutdown at %s:%d' % (self.myAddr, self.myPort)

    ######################################################
    ## Internal processing
    ######################################################
    def poll(self):
        """ This method manages ongoing training tasks.
        WARNING: this method should never block.
        It is invoked every BaseServer::poll_interval
        """
        self.pollInvokeCounter += 1
        print("poll() invoked %d times at %s for device: %s" % (self.pollInvokeCounter, self.myAddr, self.device))
        if self.pollInvokeCounter == 10:
            print("poll() invoked %d times at %s for device: %s" % (self.pollInvokeCounter, self.myAddr, self.device))

    def run(self, poll_interval=0.5):
        self.shutdownRequested = False
        while not self.shutdownRequested:
            self.handle_request()
            self.poll()
            time.sleep(poll_interval)

def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="Runtime")
    # Optional arguments for the launch helper
    parser.add_argument("--coordinatorAddr", type=str, default="localhost:12340",
                        help="IP:port to the cluster coordinator")
    parser.add_argument("--myAddr", type=str, default="localhost:1234",
                        help="IP:port this runtime should listen to."
                        "coordinator will talk to this node on this address")
    parser.add_argument("--device", type=int, default=0,
                        help="cuda device for pytorch.")
    parser.add_argument("--logdir", default=None, type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    coordinatorAddrCombined = re.split('[-:]', args.coordinatorAddr)
    coordinatorAddr = coordinatorAddrCombined[0]
    coordinatorPort = int(coordinatorAddrCombined[1])
    myAddrCombined = re.split('[-:]', args.myAddr)
    myAddr = myAddrCombined[0]
    myPort = int(myAddrCombined[1])

    runtime = Runtime(coordinatorAddr, coordinatorPort, myAddr, myPort, args.device)
    runtime.run()

if __name__ == "__main__":
    main()