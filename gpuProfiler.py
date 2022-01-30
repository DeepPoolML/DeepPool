import torch
import json
from datetime import datetime, timedelta
import deeppool_bench


class GpuProfiler:
    def __init__(self, device):
        self.device = device
        self.cache = {}

    def __saveProfile(self, path="gpuProfile.json"):
        with open(path, "w") as outfile:
            json.dump(self.cache, outfile)

    def __loadProfile(self, path="gpuProfile.json"):
        try:
            with open(path, "r") as f:
                self.cache = json.load(f)
        except IOError:
            print("[GpuProfiler] No profile file exists at %s." % path)

    def queryFwBwTime(self, layer, config):

        jitmodule = layer.scriptModule()
        inputs = layer.getRandomInputs(config[0])

        cfg = []
        ips = []
        for a in inputs:
            assert(type(a) == torch.Tensor)
            cfg.append((a.dtype, a.shape))
            ips.append(a.cuda())

        key = f"{cfg}{layer.losslayer} || {jitmodule.inlined_graph}"

        self.__loadProfile()

        if key in self.cache:
            return self.cache[key]

        fwTime, bwTime = deeppool_bench.benchmodule(jitmodule._c, inputs)

        if layer.losslayer:
            output = jitmodule.forward(*inputs).detach()
            targets = torch.zeros(output.size()[0], dtype=torch.int64).cuda()
            bwTime += deeppool_bench.benchloss(output,
                                               targets, layer.losslayer)

        self.cache[key] = (fwTime, bwTime)
        self.__saveProfile()
        return (fwTime, bwTime)
