#!/usr/bin/python3

import sys
import json
import re
import glob
import statistics
from os import listdir
from os.path import isfile, join

timetrace = [[] for i in range(10000)] # [minibatch_id] = [(time, event, rank)]
allTrace = []
maxRank = 0

def parseTrace(filePath, rank):
    global maxRank
    with open(filePath) as fin:
        count = 0
        for line in fin:
            match = re.match('(\d*\.\d+|\d+) \s+(\d+)\s+(\S+)\s*', line)
            if match:
                time = float(match.group(1))
                minibatchId = int(match.group(2))
                event = match.group(3)
                timetrace[minibatchId].append( (time, event, rank) )
                allTrace.append( (time, rank, minibatchId, event) )
                if rank > maxRank:
                    maxRank = rank
                count += 1
        print("Parsed %d iterations"%count)

def parseConfiguration(filePath):
    with open(filePath) as optimizerStdoutFile:
        parsingRegionBegan = False
        stageId = 0
        stageToRanks = {}
        stageSplitRanks = []
        nextRank = 1
        for line in optimizerStdoutFile:
            if line == "(Split start, split end) / compute time taken per stage / replication factor per stage:\n":
                parsingRegionBegan = True
                continue
            if not parsingRegionBegan:
                continue
            
            match = re.match('\((\d+), (\d+)\) (\d*\.\d+|\d+) (\d*\.\d+|\d+)', line)
            if not match:
                break
            
            start = int(match.group(1))
            end = int(match.group(2))
            computeTime = float(match.group(3))
            replicas = int(match.group(4))
            stageToRanks[stageId] = (nextRank, replicas)
            stageSplitRanks.append(nextRank)
            nextRank += replicas
            
            stageId += 1
            
    return stageSplitRanks

def analysis():
    gaps = [[] for i in  range(len(timetrace[0]))]
    printFirstTraces = 200
    minibatchId = 0
    for singleBatchTraces in timetrace:
        if len(singleBatchTraces) == 0:
            break
        if printFirstTraces > 0:
            print("MinibatchId:%d"%minibatchId)
        singleBatchTraces.sort()
        sentTime = 0.0
        prevTime = singleBatchTraces[0][0]
        for tracepoint in singleBatchTraces:
            time = tracepoint[0]
            event = tracepoint[1]
            rank = tracepoint[2]
            comment = ""
            if event.endswith("_SENT_TENSORS"):
                sentTime = time
            if event.endswith("_RECV_TENSORS") and sentTime > 0:
                comment = "NetXfer: %3d ms" % (1000 * (time - sentTime))
                sentTime = 0
            if printFirstTraces > 0:
                print("%.6f  (+%7.3f ms)  rank:%d  %22s %s"%(time, 1000 * (time - prevTime), rank, event, comment))
            prevTime = time
        printFirstTraces -= 1
        minibatchId += 1



def printAllTrace():
    allTrace.sort()
    global maxRank

    times_fwd_sent = [[] for i in range(maxRank + 1)] #FWD_SENT_TENSORS
    alltimes_fwd_xfer = [[] for i in range(maxRank + 1)]
    alltimes_bck_xfer = [[] for i in range(maxRank + 1)]
    passStartTime = [0 for i in range(maxRank + 1)]
    rankPrevEvTime = [0 for i in range(maxRank + 1)]
    fpStartTime = [0 for i in range(maxRank + 1)]
    bpStartTime = [0 for i in range(maxRank + 1)]
    lastStartTime = [0 for i in range(maxRank + 1)]
    activeTime = [0 for i in range(maxRank + 1)]
    fpTimes = [[] for i in range(maxRank + 1)]
    bpTimes = [[] for i in range(maxRank + 1)]
    iterTimes = [[] for i in range(maxRank + 1)]

    secondPortionPrev = int(allTrace[0][0]) # For drawing horizontal lines every second.
    skipMinibatchesUntil = 999999 # allTrace[0][2] + 2
    minMiniBatches = [999999 for i in range(maxRank + 1)]
    passiveBps = [False for i in range(maxRank + 1)]
    passiveFps = [False for i in range(maxRank + 1)]
    for tracepoint in allTrace:
        time = tracepoint[0]
        rank = tracepoint[1]
        minibatchId = tracepoint[2]
        event = tracepoint[3]

        if minibatchId < minMiniBatches[rank]:
            minMiniBatches[rank] = minibatchId
            skipMinibatchesUntil = max(minMiniBatches) + 2

        if event.endswith("_cpu") or minibatchId < skipMinibatchesUntil:
            continue

        # Draw a horizontal line every iter.
        if event.startswith("target_shuffle") and rank == 0:
            print("-"*230)
        suffix = ""
        # Save the time each rank started
        if event == "iter_init":
            passStartTime[rank] = time
        elif event == "target_shuffle":
            fpStartTime[rank] = time
            lastStartTime[rank] = time
        elif event == "recv_samples" and (time - fpStartTime[rank]) * 1000000 < 10:
            if not passiveFps[rank]:
                print("rank%d is passiveFP"%rank)
            passiveFps[rank] = True
            fpStartTime[rank] = 0
        elif (passiveFps[rank] and fpStartTime[rank] == 0) and event == "recv_samples_done":
            fpStartTime[rank] = time
            lastStartTime[rank] = time
        elif event == "send_samples_done_idle":
            activeTime[rank] += time - lastStartTime[rank]
            lastStartTime[rank] = 0
        elif event == "fp_done":
            if lastStartTime[rank] != 0:
                activeTime[rank] += time - lastStartTime[rank]
                lastStartTime[rank] = 0
            fpTimes[rank].append(1000000*(activeTime[rank]))
            activeTime[rank] = 0
            # fpTimes[rank].append(1000000*(time - fpStartTime[rank]))
        elif event == "bp_start":
            bpStartTime[rank] = time
            lastStartTime[rank] = time
        elif event == "bp_remainder_start" and bpStartTime[rank] == 0:
            passiveBps[rank] = True
        elif (passiveBps[rank] and bpStartTime[rank] == 0) and event == "recv_samples_done":
            bpStartTime[rank] = time
            lastStartTime[rank] = time
        elif event == "bp_done":
            if lastStartTime[rank] != 0:
                activeTime[rank] += time - lastStartTime[rank]
                lastStartTime[rank] = 0
            # bpTimes[rank].append(1000000*(time - bpStartTime[rank]))
            bpTimes[rank].append(1000000*(activeTime[rank]))
            activeTime[rank] = 0

            iterTimes[rank].append(1000000*(time - fpStartTime[rank]))
            suffix = "rank: %d  fp: %4.f bp: %4.f iter: %4.f" % (rank, fpTimes[rank][-1], bpTimes[rank][-1], iterTimes[rank][-1])
            bpStartTime[rank] = 0
            passiveBps[rank] = False

        strBuilder = "%.6f    %6.f us  (+%6.f us)  " % (time, (time - passStartTime[0]) * 1000000, (time - rankPrevEvTime[rank])* 1000000)
        for r in range(maxRank+1):
            # strBuilder += "|" if (r+1) in stageSplitRanks else " "
            # stage = rankToStage[r]
            # strBuilder += splitStr[stage] if (r+1) in stageSplitRanks else " "
            strBuilder += "[%2d] %-20s"%(minibatchId, event) if r == rank else " " * 25
        # strBuilder += commStatStr
        rankPrevEvTime[rank] = time
        print(strBuilder + suffix)

    print("\n#               ", end="")
    for rank in range(maxRank+1):
        print("          RANK: %2d                        | " % rank, end="")    
    print("\n#   Type        ", end="")
    for rank in range(maxRank+1):
        print("   avg  ( min / median /   max / stdev)   | ", end="")
    print("\n# --------------" + "--------------------------------------------"*(maxRank + 1))
    print("  fpTime (us):  ", end="")
    for rank in range(maxRank+1):
        avg = sum(fpTimes[rank]) / len(fpTimes[rank])
        first100avg = sum(fpTimes[rank][:100]) / 100
        last100avg = sum(fpTimes[rank][-100:]) / 100
        fpTimes[rank].sort()
        minimum = fpTimes[rank][0]
        maximum = fpTimes[rank][-1]
        median = fpTimes[rank][int(len(fpTimes[rank])/2)]
        std = statistics.stdev(fpTimes[rank])
        # print(" %6.f <%5.f/%5.f> (%5.f/%5.f/%5.f)   | " % (avg, first100avg, last100avg, minimum, median, maximum), end="")
        print(" %6.f (%5.f / %5.f / %5.f / %5.f)   | " % (avg, minimum, median, maximum, std), end="")

    print("\n  bpTime (us):  ", end="")
    for rank in range(maxRank+1):
        avg = sum(bpTimes[rank]) / len(bpTimes[rank])
        first100avg = sum(bpTimes[rank][:100]) / 100
        last100avg = sum(bpTimes[rank][-100:]) / 100
        bpTimes[rank].sort()
        minimum = bpTimes[rank][0]
        maximum = bpTimes[rank][-1]
        median = bpTimes[rank][int(len(bpTimes[rank])/2)]
        std = statistics.stdev(bpTimes[rank])
        print(" %6.f (%5.f / %5.f / %5.f / %5.f)   | " % (avg, minimum, median, maximum, std), end="")

    print("\n fp + bp (us):  ", end="")
    gpuMsecAvgSum = 0
    gpuMsecP50Sum = 0
    for rank in range(maxRank+1):
        times = [fpt + bpt for (fpt, bpt) in zip(fpTimes[rank], bpTimes[rank])]
        avg = sum(times) / len(times)
        gpuMsecAvgSum += avg
        first100avg = sum(times[:100]) / 100
        last100avg = sum(times[-100:]) / 100
        times.sort()
        minimum = times[0]
        maximum = times[-1]
        median = times[int(len(times)/2)]
        gpuMsecP50Sum += median
        std = statistics.stdev(times)
        print(" %6.f (%5.f / %5.f / %5.f / %5.f)   | " % (avg, minimum, median, maximum, std), end="")

    print(" gpuMsec ==> avg: %d, p50: %d" % (gpuMsecAvgSum, gpuMsecP50Sum), end="")

    print("\niterTime (us):  ", end="")
    for rank in range(maxRank+1):
        avg = sum(iterTimes[rank]) / len(iterTimes[rank])
        first100avg = sum(iterTimes[rank][:100]) / 100
        last100avg = sum(iterTimes[rank][-100:]) / 100
        iterTimes[rank].sort()
        minimum = iterTimes[rank][0]
        maximum = iterTimes[rank][-1]
        median = iterTimes[rank][int(len(iterTimes[rank])/2)]
        std = statistics.stdev(iterTimes[rank])
        print(" %6.f (%5.f / %5.f / %5.f / %5.f)   | " % (avg, minimum, median, maximum, std), end="")
                
    # print("="*230)
    # strBuilder = "%17s  " % "fwd_avg"
    # for r in range(maxRank+1):
    #     alltimes_fwd_xfer[r] = alltimes_fwd_xfer[r][1:] # exclude the first value.
    #     strBuilder += "    | >>  " if (r+1) in stageSplitRanks else " "
    #     avg = sum(alltimes_fwd_xfer[r]) / len(alltimes_fwd_xfer[r]) if len(alltimes_fwd_xfer[r]) != 0 else 0
    #     maximum = max(alltimes_fwd_xfer[r]) if len(alltimes_fwd_xfer[r]) != 0 else 0
    #     minimum = min(alltimes_fwd_xfer[r]) if len(alltimes_fwd_xfer[r]) != 0 else 0
    #     print(alltimes_fwd_xfer[r])
    #     strBuilder += "%4d/%4d/%4d " % (avg, minimum, maximum)
    # print(strBuilder)
    # strBuilder = "%17s  " % "bck_avg"
    # for r in range(maxRank+1):
    #     alltimes_bck_xfer[r] = alltimes_bck_xfer[r][1:] # exclude the first value.
    #     strBuilder += " << |     " if (r+1) in stageSplitRanks else " "
    #     avg = sum(alltimes_bck_xfer[r]) / len(alltimes_bck_xfer[r]) if len(alltimes_bck_xfer[r]) != 0 else 0
    #     minimum = min(alltimes_bck_xfer[r]) if len(alltimes_bck_xfer[r]) != 0 else 0
    #     maximum = max(alltimes_bck_xfer[r]) if len(alltimes_bck_xfer[r]) != 0 else 0
    #     print(alltimes_bck_xfer[r])
    #     strBuilder += "%4d/%4d/%4d " % (avg, minimum, maximum)
    # print(strBuilder)


def main():
    if len(sys.argv) != 3:
        print("Wrong number of arguments!")
        print("Usage: parseResults.py folderPath gpuCount")
        return

    gpuCount = int(sys.argv[2])
    # stageSplitRanks = parseConfiguration(sys.argv[1] + "runtime%d.txt"%gpuCount)
    folderPath = sys.argv[1]
    if sys.argv[1][-1] != '/':
        folderPath += '/'

    for filename in listdir(folderPath):
        match = re.match("runtime(\d+).out", filename)
        if not match:
            continue
        
        print("Matched! %s"%filename)
        rank = int(match.group(1))
        parseTrace(folderPath + filename, rank)
    
    # analysis()
    printAllTrace()
        
if __name__ == "__main__":
    main()