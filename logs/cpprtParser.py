#!/usr/bin/env python

import re
import sys

data = {}

def scan(f):
    global data
    lineCount = 0
    for line in f:
        lineCount += 1
        match = re.match('cpprt([0-9]+)\.out.* job (\S+) is completed \([0-9.]+ iters, ([0-9.]+) ms/iter, ([0-9.]+) iter/s,\s+([0-9.]+) be.*', line)
        match2 = re.match('.* job (\S+) is completed \([0-9.]+ iters, ([0-9.]+) ms/iter, ([0-9.]+) iter/s,\s+([0-9.]+) be.*', line)
        if match:
            rank = int(match.group(1))
            job = match.group(2)
            iterMs = float(match.group(3))
            tput = float(match.group(4))
            beTput = float(match.group(5))
        elif match2:
            job = match2.group(1)
            iterMs = float(match2.group(2))
            tput = float(match2.group(3))
            beTput = float(match2.group(4))
        else:
            continue
            # print("no match! : ", line)
            
        if job not in data:
            data[job] = {"count": 0, "iterMs": 0.0, "iterMsMax": 0.0, "tput": 0, "beTput": 0}
        data[job]["count"] += 1
        data[job]["iterMs"] += iterMs
        data[job]["iterMsMax"] = max(data[job]["iterMsMax"], iterMs)
        data[job]["tput"] += tput
        data[job]["beTput"] += beTput
        # print("rank%2d: %20s %8.2f %8.2f %8.2f" % (rank, job, iterMs, tput, beTput))
    # print("file length: ", lineCount)

def printOut():
    global data
    print("#                Job #GPUs maxIterMs avgIterMs   fgTput     bgTput")
    for job in data:
        c = data[job]["count"]
        print("%20s    %2d  %8.2f  %8.2f %8.2f %10.1f" % (job, c, data[job]["iterMsMax"], data[job]["iterMs"] / c, data[job]["tput"] / c, data[job]["beTput"]))


if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) == 2:
        scan(open(sys.argv[1]))
    else:
        for filename in ["cpprt%d.out" % r for r in range(8)]:
            print("# Scanning ", filename)
            scan(open(filename))
    printOut()

        
# cpprt4.out:1633130141.261867690 taskManager.cpp:295 in poll NOTICE[1]: A training job vgg16_8_16__DP is completed (1900 iters, 11.23 ms/iter, 89.07 iter/s, 763.52 be img/s). AverageTiming (ms) => zero: 0.0, load:0.0, fp:0.0, loss:0.0, bp:0.0, opt: 0.0, iter:0.0 P50 (ms) => fp:0.0, loss:0.0, bp:0.0, iter:0.0