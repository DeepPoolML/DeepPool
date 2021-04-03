import time
import collections
import torch

class EventTypes:
    # EventName       ID (will be assigned by Timetrace.buildTablesForEventTypes)
    iter_init               = None
    target_shuffle          = None
    # fp_start                = None
    fp_done                 = None
    recv_samples            = None
    recv_samples_done       = None
    recv_samples_cpu        = None
    recv_samples_done_cpu   = None
    send_samples            = None
    send_samples_done       = None
    send_samples_done_idle  = None
    bp_start                = None
    bp_remainder_start      = None
    bp_done                 = None
    # bp_recv_samples         = None
    # bp_recv_samples_done    = None
    # bp_send_smples          = None
    # bp_send_smples_done     = None
    # Add more events here.

class Timetrace:
    # traces = [] # (time, eventId, minibatch_id)
    traces = collections.deque(maxlen=5000)
    # strToEid, eidToStr = Timetrace.buildTablesForEventTypes()
    cudaEvents = []
    iterStartCpuTime = None
    iterMinibatchId = None

    @classmethod
    def buildTablesForEventTypes(cls):
        strToEid = {}
        eidToStr = {}
        nextId = 1
        for k, oldV in vars(EventTypes).items():
            if not k.startswith("__"):
                v = nextId
                nextId += 1
                strToEid[k] = v
                eidToStr[v] = k
                setattr(EventTypes, k, v)
        cls.strToEid = strToEid
        cls.eidToStr = eidToStr

    @classmethod
    def record(cls, eid, minibatch_id=None, comment = None):
        if minibatch_id == None:
            minibatch_id = cls.iterMinibatchId
        cls.traces.append((time.time(), eid, minibatch_id, comment))
    
    @classmethod
    def cudaInitIter(cls, minibatch_id):
        # synchronize cuda. take cpu time, insert cuda event for initial stuff.. 
        if len(cls.cudaEvents) != 0:
            raise Exception("[Timetrace.cudaInitIter] cudaEvents list is not empty.")
        torch.cuda.synchronize()
        cls.iterStartCpuTime = time.time()
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        cls.cudaEvents.append( (ev_start, EventTypes.iter_init, None) )
        cls.iterMinibatchId = minibatch_id
    
    @classmethod
    def cudaFinishIter(cls):
        # synchronize cuda. flush all timing data.
        torch.cuda.synchronize()
        if len(cls.cudaEvents) <= 1:
            cls.cudaEvents.clear()
            return
        
        # Flush all cuda events to traces.
        startEvent, startEid, startComment = cls.cudaEvents[0]
        cls.traces.append((cls.iterStartCpuTime, startEid, cls.iterMinibatchId, startComment))
        for i in range(1, len(cls.cudaEvents)):
            cudaEv, eid, comment = cls.cudaEvents[i]
            cudaElapsedMs = startEvent.elapsed_time(cudaEv)
            absTime = cls.iterStartCpuTime + (cudaElapsedMs / 1000.)
            cls.traces.append((absTime, eid, cls.iterMinibatchId, comment))

            cls.iterMinibatchId

        cls.cudaEvents.clear()

    @classmethod
    def cudaRecord(cls, eid, comment = None):
        assert len(cls.cudaEvents) > 0 # cudaInitIter should be called before.
        # STOPPED HERE. test with record() first...
        # TODO for cuda.. register cuda event. push back cuda events..
        # flush method which does cuda.synchronize  take cpu time. compute absolute time for all cuda events.
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        cls.cudaEvents.append( (ev, eid, comment) )

    @classmethod
    def printAllTraces(cls):
        print("# timeInSec        batch             event          comment")
        for (time, eid, minibatch_id, comment) in cls.traces:
            if comment is None:
                comment = ""
            if minibatch_id is None:
                minibatch_id = 0
            print("%5.6f  %5d   %s  %s" % (time, minibatch_id, cls.eidToStr[eid], comment))

    @classmethod
    def printStats(cls, skip_initial_minibatch = 20):
        gaps = [[] for i in  range(len(cls.eidToStr))]
        prevTime = cls.traces[0][0]
        for (time, eid, minibatch_id, comment) in cls.traces:
            elapsed = time - prevTime
            prevTime = time

            if minibatch_id > skip_initial_minibatch:
                gaps[eid].append(elapsed)
        
        print("# Printing time elapsed for each event. skipped first %d minibatches" % skip_initial_minibatch)
        print("# Event             Avg (ms)    Median (ms)   Min (ms)    Max (ms)")
        for i in range(1, len(cls.eidToStr)):
            avg = sum(gaps[i]) / len(gaps[i])
            median = gaps[i][int(len(gaps[i]) / 2)]
            print("%22s %7.3f %7.3f %7.3f %7.3f" % (cls.eidToStr[i], avg, median, min(gaps[i]), max(gaps[i])))

Timetrace.buildTablesForEventTypes()
# print(Timetrace.strToEid)
# print(Timetrace.eidToStr)