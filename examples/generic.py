import warnings
import torch
import time
# from typing import Callable, Any, Optional, Tuple, List
import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from parallelizationPlanner import CostSim
from clusterClient import ClusterClient
from jobDescription import TrainingJob

import json
import re

"""
nm = {
    "aten::_convolution": "conv2d",
    "aten::batch_norm": "layerNorm",
    "aten::relu": "ReLU2d",
    "aten::relu_": "ReLU2d",
    "aten::max_pool2d": "maxPool2d",
    "aten::add": "aten::add",
    "aten::adaptive_avg_pool2d": "adAvgPool2d",
    "aten::linear": "linear",
    "aten::dropout": "dropout",
    "aten::avg_pool2d": "avgPool2d",
    "aten::cat": "concat",

# vit
    "aten::reshape": "reshape",
    "aten::repeat": "repeat",
    "aten::einsum": "einsum",
    "aten::mul": "mul",
    "aten::layer_norm": "layerNorm",
    "aten::gelu": "gelu",
    "aten::softmax": "softmax",
}
"""

def extract(pm):
    rex = "t?\[(.*)\], "
    lns = re.match(rex, pm).group(1)
    lns = lns.replace(",", "")
    return list(map(int, lns.split()))

def load_model_set(cs):
    with open(MODELLOC) as f:
        dat = json.loads(f.read())

    layers = []

    for li, layer in enumerate(dat):
        module = torch.jit.load(layer["mod_file"])
        module.cpu()
        prevls = []
        for i in layer["deps"]:
            if i == -1: continue
            prevls.append(layers[i])

        old_cfg = {a[0]: a[1] for a in layer["arguments"]}
        new_cfg = {}
        xname = layer["name"] #nm[layer["name"]] 

        new_cfg["ext_iput"] = extract(layer["iputs"][0])[1:]

        if "convolution" in layer["name"]:
            inputdims = extract(old_cfg["input"])
            new_cfg["in_channels"] = inputdims[1]

            weights = extract(old_cfg["weight"])
            if weights[2] == weights[3]: new_cfg["kernel_size"] = weights[2]
            else: new_cfg["kernel_size"] = tuple(weights[2:])


            for fname in ["padding", "stride"]:
                if not fname in old_cfg: continue
                x = extract(old_cfg[fname])
                new_cfg[fname] = x[0] if x[0] == x[1] else tuple(x)

            outs = [a[1] for a in layer["returns"]]
            assert len(outs) == 1
            nxtidims = extract(outs[0])
            new_cfg["out_channels"] = nxtidims[1]

        if "relu" in layer["name"]:
            dims = extract(old_cfg["self"])
            # xname = "ReLU1d" if len(dims) < 3 else "ReLU2d"

        if "adaptive_avg_pool2d" in layer["name"]:
            output_size = extract(old_cfg["output_size"])
            new_cfg.update({"output_width": output_size[0], "output_height": output_size[1]})

        if "linear" in layer["name"]:
            print(old_cfg)
            inputdims = extract(old_cfg["input"])
            ws = extract(old_cfg["weight"])
            assert(inputdims[-1] == ws[1])
            new_cfg.update({"in_features": ws[1], "out_features": ws[0]})

        if "max_pool2d" in layer["name"] or xname == "avgPool2d":
            for fname in ["kernel_size", "padding", "stride", "dilation"]:
                if not fname in old_cfg: continue
                x = extract(old_cfg[fname])
                new_cfg[fname] = x[0] if x[0] == x[1] else tuple(x)

        l = cs.GeneralLayer(
            module, xname, new_cfg, mustTrace=False, custom_previous_layers=prevls)
        l.moduleSavedLocation = layer["mod_file"]
        layers.append(l)

def main(gpuCount, globalBatch, amplificationLimit=2.0, dataParallelBaseline=False, netBw=2.66E5, spatialSplit=False, simResultFilename=None, simOnly=False, use_be=False):
    cs = CostSim(None, netBw=netBw, verbose=True, gpuProfileLoc="inceptionLayerGpuProfileA100.txt") #, gpuProfileLocSub="inceptionLayerGpuProfileA100.txt")
    load_model_set(cs)
    cs.printAllLayers(slient=False)
    cs.computeInputDimensions((3,299,299))
    # job, iterMs, gpuMs = cs.searchBestSplits(gpuCount, globalBatch, amplificationLimit=amplificationLimit, dataParallelBaseline=dataParallelBaseline, spatialSplit=spatialSplit)
    cs.to_dot(simResultFilename, globalBatch, justdag=True)

    # if dataParallelBaseline:
    #     dpIterUsec, dpFpUsec, dpBpUsec = profiler.benchModel(model, (3, 299, 299), int(globalBatch / gpuCount))
    #     print("(DP baseline) whole model bench: %.1f ms (fp: %.1f, bp: %.1f)" % (dpIterUsec / 1000, dpFpUsec / 1000, dpBpUsec / 1000))

    # cs.JustDoDP(gpuCount, globalBatch)
    # exit(0)
    job, iterMs, gpuMs, maxGpusUsed = cs.JustDoDP(gpuCount, globalBatch) #cs.searchBestSplitsV3(gpuCount, globalBatch, amplificationLimit=amplificationLimit, dataParallelBaseline=True, spatialSplit=spatialSplit)
    print("  %2d    %2d   %4.1f  %4.1f\n" % (globalBatch, maxGpusUsed, iterMs, gpuMs))
    cs.to_dot(simResultFilename, globalBatch)
    # cs.to_gpuTimeline("Inception v3, Burst Parallel", maxGpusUsed, dataParallelBaseline)
    jobInJson = job.dumpInJSON()

    # for rank in range(gpuCount):
    #     print("GPU rank: %d"%rank)
    #     print(job.dumpSingleRunnableModule(rank))

    job2 = TrainingJob("test", None, None, 0, 0, "")
    job2.loadJSON(jobInJson)
    assert(jobInJson == job2.dumpInJSON())
    print("Load/Dump returned the same output? %s" % ("true" if jobInJson == job2.dumpInJSON() else "false"))

    jobs = [job2.dumpSingleRunnableModule(rank) for rank in range(job2.getGpusUsed())]
    # for j in jobs:
        # from pprint import pprint
        # pprint(json.loads(j))
    # exit(0)
    # print(jobInJson)
    
    if maxGpusUsed > 8:
        print("maxGpusUsed: ", maxGpusUsed, " is bigger than 8. Can't schedule this job.")
        exit(-1)
    
    if not spatialSplit and not simOnly:
        cc = ClusterClient()
        jobName = f"Generic_%s_%d_%d_%2.1f%s" % (MODELLOC.split("/")[-1].split(".")[0], gpuCount, globalBatch, amplificationLimit, "_DP" if dataParallelBaseline else "")
        jobName += "_BE" if use_be else ""
        cc.submitTrainingJob(jobName, jobInJson, use_be)

    if simResultFilename != None:
        f = open(simResultFilename, "a")
        f.write("  %2d    %2d   %4.1f  %4.1f\n" % (globalBatch, maxGpusUsed, iterMs, gpuMs))
        f.close()

        if gpuCount == 8:
            f = open(simResultFilename, "r")
            print(f.read())
            f.close()


def runAllConfigs(modelName: str, clusterType: str, simOnly=True):
    if clusterType == "V100":
        netBw = 22937
    elif clusterType == "A100":
        netBw = 2.66E5
    elif clusterType == "10Gbps":
        netBw = 1.25E3
    elif clusterType == "100Gbps":
        netBw = 1.25E4
    elif clusterType == "10Tbps":
        netBw = 1.25E6
    else:
        print("Wrong cluster type. Put either V100 or A100")

    gpuCounts = [1, 2, 4, 8]
    # gpuCounts = [1, 2, 4]
    globalBatchSize = 32
    # globalBatchSize = 16
    # globalBatchSize = 8
    limitAndBaseline = [(2.0, True, False), (1.5, False, False), (2.0, False, False), (2.5, False, False)]
    # limitAndBaseline = [(99, False, True)]
    # limitAndBaseline = []
    for lim, baseline, spatialSplit in limitAndBaseline:
        simResultFilename = "%s_%s_b%d_lim%2.1f_sim.data" % (modelName, "DP" if baseline else "MP", globalBatchSize, lim)
        f = open(simResultFilename, "w")
        f.write("#batch GPUs IterMs  GpuMs\n")
        f.close()

        for gpuCount in gpuCounts:
            if not simOnly:
                preSize = os.stat('runtimeResult.data').st_size
            main(gpuCount, globalBatchSize, amplificationLimit=lim, dataParallelBaseline=baseline, netBw=netBw, spatialSplit=spatialSplit, simResultFilename=simResultFilename, simOnly=simOnly)
            # check exp finished.
            if not simOnly:
                print("runtimeResult.data's original size: ", preSize)
                while os.stat('runtimeResult.data').st_size == preSize and not spatialSplit:
                    time.sleep(10)
                print("runtimeResult.data's current size: ", os.stat('runtimeResult.data').st_size)
        
        if not spatialSplit and not simOnly:
            fw = open("%s_%s_b%d_lim%2.1f_run.data" % (modelName, "DP" if baseline else "MP", globalBatchSize, lim), "w")
            fr = open('runtimeResult.data', "r")
            fw.write("#batch GPUs IterMs  GpuMs\n")
            fw.write(fr.read())
            fw.close()
            fr.close()

        fr = open('runtimeResult.data', "w")
        fr.close()

def runStrongScalingBench():
    global cs
    netBw = 2.66E5
    cs = CostSim(None, netBw=netBw, verbose=False)
    inputSize = (3,299,299)
    model = Inception3(aux_logits=False)

    fakeInputSize = (16,3,299,299)
    fakeInput = torch.zeros(fakeInputSize)
    traced = torch.jit.trace(model, fakeInput)
    torch.jit.save(traced, "modules/inception.pt")
    
    print("Model: ", "Inception3")
    print("BatchSize  iterMs    fpMs    bpMs")
    for batchSize in [2 ** exp for exp in range(1, 9)]:
        assert False
        # iterTime, fpTime, bpTime = profiler.benchModel(model, inputSize, batchSize)
        # print(" %8d  %6.1f  %6.1f  %6.1f" %
        #    (batchSize, iterTime / 1000, fpTime / 10000, bpTime / 1000))

if __name__ == "__main__":

    global MODELLOC
    assert(len(sys.argv) > 1)
    MODELLOC = sys.argv[1]

    if len(sys.argv) == 4:
        gpuCount = int(sys.argv[2])
        globalBatchSize = int(sys.argv[3])
        simResultFilename = "%s_%s_b%d_sim.data" % ("inception", "DP", globalBatchSize)
        main(gpuCount, globalBatchSize, dataParallelBaseline=True)
    elif len(sys.argv) >= 5:
        use_be = len(sys.argv) > 5 and int(sys.argv[5]) == 1
        gpuCount = int(sys.argv[2])
        globalBatchSize = int(sys.argv[3])
        # simResultFilename = "%s_%s_b%d_lim%2.1f_sim.data" % ("inception", "MP", globalBatchSize, amplificationLimit)
        if sys.argv[4] == "DP":
            main(gpuCount, globalBatchSize, dataParallelBaseline=True, use_be=use_be)
        else:
            amplificationLimit = float(sys.argv[4])
            main(gpuCount, globalBatchSize, amplificationLimit, use_be=use_be)
            # main(gpuCount, globalBatchSize, amplificationLimit, simResultFilename = simResultFilename, use_be=use_be)
    elif len(sys.argv) == 3:
        print("Run all configs")
        runAllConfigs("inceptionV3", sys.argv[2])
    elif len(sys.argv) == 2:
        runStrongScalingBench()
    else:
        print("Wrong number of arguments.\nUsage: ")
