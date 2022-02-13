import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss


#from torchsummary import summary
import torch
from torch import nn, Tensor
import numpy as np
from einops import rearrange
import random
from typing import List, Tuple
from einops import repeat
import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
# sys.path.append('/transformers/src/transformers/models/gpt2')
sys.path.append(os.path.abspath('../../transformers'))

from parallelizationPlanner import CostSim
from clusterClient import ClusterClient
from jobDescription import TrainingJob
# from transformers import GPT2Model

# from Project1.file1 import something

from transformers.models.gpt2 import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, GPT2Model
#from datasets import load_dataset

def main(gpuCount, globalBatch, amplificationLimit=2.0, dataParallelBaseline=False, netBw=2.66E5, spatialSplit=False, simResultFilename=None, simOnly=False, use_be=False):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    config = GPT2Config()

    global cs
    # cs = CostSim(profiler, netBw=netBw, verbose=True)
    cs = CostSim(None, netBw=netBw, verbose=True, gpuProfileLoc="gpt2LayerGpuProfileA100V2.txt")
    # model = ViT(img_dim=256, in_channels=3, patch_dim=16, num_classes=1000, dim=512, blocks=12, dim_linear_block=3072)
    # model = ViT(img_dim=256, in_channels=3, patch_dim=32, num_classes=1000, dim=1024, blocks=24, heads=16, dim_linear_block=4092)
    # model = GPT2Model.from_pretrained('gpt2')
    config = GPT2Config()
    config.n_layer = 1
    config.n_head = 16
    model = GPT2LMHeadModel(config, cs)

    # dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<|endoftext|>')
    # # tokenizer.pad_token = tokenizer.eos_token
    # dataset = dataset.map(lambda t: tokenizer(t['text'], truncation=True, padding='max_length'), batched=True, batch_size=8, drop_last_batch=True)
    # dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    # batch = next(iter(dataloader))
    # # model = model.to('cuda')
    # model(**batch)

    cs.printAllLayers(slient=False)
    cs.computeInputDimensions((1024,), dtype=torch.int32)
    cs.setLossFunction("CrossEntropyLoss")
    # job, iterMs, gpuMs = cs.searchBestSplits(gpuCount, globalBatch, amplificationLimit=amplificationLimit, dataParallelBaseline=dataParallelBaseline, spatialSplit=spatialSplit)

    # if dataParallelBaseline:
    #     dpIterUsec, dpFpUsec, dpBpUsec = profiler.benchModel(model, (3, 299, 299), int(globalBatch / gpuCount))
    #     print("(DP baseline) whole model bench: %.1f ms (fp: %.1f, bp: %.1f)" % (dpIterUsec / 1000, dpFpUsec / 1000, dpBpUsec / 1000))

    if dataParallelBaseline:
        job, iterMs, gpuMs, maxGpusUsed = cs.JustDoDP(gpuCount, globalBatch)
    else:
        job, iterMs, gpuMs, maxGpusUsed = cs.searchBestSplitsV3(gpuCount, globalBatch, amplificationLimit=amplificationLimit, dataParallelBaseline=dataParallelBaseline, spatialSplit=spatialSplit)
    print("  %2d    %2d   %4.1f  %4.1f\n" % (globalBatch, maxGpusUsed, iterMs, gpuMs))
    cs.to_dot(simResultFilename, globalBatch)
    # cs.to_gpuTimeline("Inception v3, Burst Parallel", maxGpusUsed, dataParallelBaseline)
    jobInJson = job.dumpInJSON()

    # for rank in range(maxGpusUsed):
    #     print("GPU rank: %d"%rank)
    #     print(job.dumpSingleRunnableModule(rank))

    # job2 = TrainingJob("test", None, None, 0, 0, "")
    # job2.loadJSON(jobInJson)
    # assert(jobInJson == job2.dumpInJSON())
    # print("Load/Dump returned the same output? %s" % ("true" if jobInJson == job2.dumpInJSON() else "false"))
    # print(jobInJson)
    
    job2 = TrainingJob("test", None, None, 0, 0, "")
    job2.loadJSON(jobInJson)
    assert(jobInJson == job2.dumpInJSON())
    print("Load/Dump returned the same output? %s" % ("true" if jobInJson == job2.dumpInJSON() else "false"))
    

    if maxGpusUsed > 8:
        print("maxGpusUsed: ", maxGpusUsed, " is bigger than 8. Can't schedule this job.")
        return
    
    if not spatialSplit and not simOnly:
        cc = ClusterClient()
        jobName = "gpt2_%d_%d_%2.1f%s" % (gpuCount, globalBatch, amplificationLimit, "_DP" if dataParallelBaseline else "")
        jobName += "_BE" if use_be else ""
        cc.submitTrainingJob(jobName, jobInJson, runbe=use_be)

def runStrongScalingBench():
    global cs
    netBw = 2.66E5
    cs = CostSim(None, netBw=netBw, verbose=False)
    inputSize = (1024)
    # model = ViT(img_dim=256, in_channels=3, patch_dim=16, num_classes=1000, dim=512)
    config = GPT2Config()
    config.n_layer = 1
    config.n_head = 16
    config.use_cache=True
    model = GPT2LMHeadModel(config)

    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<|endoftext|>')
    # tokenizer.pad_token = tokenizer.eos_token
    dataset = dataset.map(lambda t: tokenizer(t['text'], truncation=True, padding='max_length'), batched=True, batch_size=8, drop_last_batch=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    batch = next(iter(dataloader))
    model = model.to('cuda')
    # model()
    #summary(model, [tuple(batch['input_ids'].size()), tuple(batch['attention_mask'].size())])

    traced = torch.jit.trace(model, (batch['input_ids'].to('cuda'), batch['attention_mask'].to('cuda')))

    model.to('cpu')
    batch['input_ids'].to('cpu')
    batch['attention_mask'].to('cpu')
    torch.jit.save(traced, "modules/gpt2.pt")
    
    print("Model: ", "GPT2(256, 3, 16, 1000, 512")
    print("BatchSize  iterMs    fpMs    bpMs")
    for batchSize in [2 ** exp for exp in range(1, 5)]:
        assert False
        # iterTime, fpTime, bpTime = profiler.benchModelNLP(model, inputSize, batchSize, profile=True)
        # print(" %8d  %6.1f  %6.1f  %6.1f" %
            # (batchSize, iterTime / 1000, fpTime / 10000, bpTime / 1000))


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        gpuCount = int(sys.argv[1])
        globalBatchSize = int(sys.argv[2])
        use_be = len(sys.argv) > 4 and int(sys.argv[4]) == 1
        if len(sys.argv) < 4 or sys.argv[3] == "DP":
            simResultFilename = "%s_%s_b%d_sim.data" % ("gpt2", "DP", globalBatchSize)
            main(int(sys.argv[1]), int(sys.argv[2]), dataParallelBaseline=True, simResultFilename=simResultFilename, use_be=use_be)
        else:
            ampLimit = float(sys.argv[3])
            simResultFilename = "%s_%s_b%d_lim%2.1f_sim.data" % ("gpt2", "MP", globalBatchSize, ampLimit)
            main(int(sys.argv[1]), int(sys.argv[2]), amplificationLimit=ampLimit, simResultFilename=simResultFilename, use_be=use_be)
    elif len(sys.argv) == 2:
        print("Run all configs")
        # runAllConfigs("gpt2", sys.argv[1])
    elif len(sys.argv) == 1:
        runStrongScalingBench()
    else:
        print("Wrong number of arguments.\nUsage: ")
