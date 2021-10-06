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
import sys
import time
import xmlrpc.client

PRIVATE_ADDR_FILENAME="aws_ec2_tools/aws-started-privateIps.txt"
COORD_PORT = 12347

class ClusterClient:
    """ A handle to submit training job to cluster. """

    def __init__(self, coordinatorAddr: str = None, coordinatorPort: int = None, maxRetries = 5):
        if coordinatorAddr == None:
            coordinatorAddr = "127.0.0.1"
            print("[ClusterClient] auto filled coodinator address: ", coordinatorAddr)

            # with open(PRIVATE_ADDR_FILENAME, "r") as f:
            #     privateIps = []
            #     for line in f:
            #         privateIps.extend(line.split())
            #     coordinatorAddr = privateIps[0]
            #     print("[ClusterClient] auto filled coodinator address: ", coordinatorAddr)
        if coordinatorPort == None:
            coordinatorPort = COORD_PORT
            print("[ClusterClient] auto filled coodinator port: ", coordinatorPort)

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
    
    def submitTrainingJob(self, jobName: str, trainingJobInJSON: str, runbe=False):
        f = open(jobName + ".json", "w")
        f.write(trainingJobInJSON)
        f.close()
        print("Saved the training job at: " + jobName + ".json")
        self.proxy.scheduleTraining(jobName, trainingJobInJSON, runbe)

    def submitSavedTrainingJob(self, path, runbe=False):
        f = open(path, "r")
        trainingJobInJSON = f.read()
        f.close()
        self.proxy.scheduleTraining(path[:-5], trainingJobInJSON, runbe)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        cc = ClusterClient()
        cc.submitSavedTrainingJob(sys.argv[1])