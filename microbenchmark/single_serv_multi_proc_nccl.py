import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from argparse import ArgumentParser
import torch.cuda.profiler as profiler
import torch.cuda.nvtx as nvtx
import pyprof
# pyprof.init()

PRIVATE_ADDR_FILENAME="../aws_ec2_tools/aws-started-privateIps.txt"

def run(rank, size):
	# tensor = torch.zeros(1)
	with torch.autograd.profiler.emit_nvtx():
		num_iters = 30500
		print('num of comms: ', num_iters)
		start_pt = [20000] # [20, 1020, 10020, 20020, 30020]
		end_pt = [20100] # [40, 1040, 10040, 20040, 30040]

		# tensorSizes = [(1, 64, 224, 224), (2, 64, 224, 224), (4, 64, 224, 224), (16, 64, 112, 112), (16, 16, 224, 224)]
		tensorSizes = [(1, 16, 112, 112), (1, 64, 112, 112), (1, 64, 224, 224), (2, 64, 224, 224), (4, 64, 224, 224), (16, 64, 112, 112), (16, 16, 224, 224)]
		if rank == 0:
			for tensorSize in tensorSizes:
				tensorReadyTimes = []
				sentTimes = []
				for i in range(num_iters):
					# tensor += 1
					ev_start = torch.cuda.Event(enable_timing=True)
					ev_start.record()
					tensor = torch.empty(tensorSize, dtype=torch.float, device=torch.device('cuda', torch.cuda.current_device()))
					# tensor.fill_(i)
					ev_tensorReady = torch.cuda.Event(enable_timing=True)
					ev_tensorReady.record()
					dist.send(tensor=tensor, dst=1, tag=i)
					ev_sent = torch.cuda.Event(enable_timing=True)
					ev_sent.record()
					ev_sent.synchronize()
					tensorReadyTimes.append(ev_start.elapsed_time(ev_tensorReady))
					sentTimes.append(ev_tensorReady.elapsed_time(ev_sent))
					if i in start_pt:
						torch.cuda.synchronize()
						tic = time.perf_counter()
						start_idx = i
						profiler.start()
					if i in end_pt:
						torch.cuda.synchronize()
						toc = time.perf_counter()
						end_idx = i
						dif = toc-tic
						profiler.stop()
						tensorReadyTimes = tensorReadyTimes[20000:]
						sentTimes = sentTimes[20000:]
						print("tensorSize: ", tensorSize, " bytes: ", tensor.element_size()*tensor.nelement())
						print('measure starts from ', start_idx, '-th comm')
						print('tic-toc total (s): ', dif, ' for ', end_idx-start_idx, ' iterations')
						print('average per send-recv comm (ms): ', dif*1000/(end_idx-start_idx))
						print("tneosrReadyTime: %.3f ms" % (sum(tensorReadyTimes)/ len(tensorReadyTimes)))
						print("SentTime: %.3f ms" % (sum(sentTimes)/ len(sentTimes)))
		else:
			for tensorSize in tensorSizes:
				tensorReadyTimes = []
				sentTimes = []
				for i in range(num_iters):
					# ev_start = torch.cuda.Event(enable_timing=True)
					# ev_start.record()
					tensor = torch.empty(tensorSize, dtype=torch.float, device=torch.device('cuda', torch.cuda.current_device()))
					# ev_tensorReady = torch.cuda.Event(enable_timing=True)
					# ev_tensorReady.record()
					dist.recv(tensor=tensor, src=0, tag=i)
					ev_sent = torch.cuda.Event(enable_timing=True)
					# ev_sent.record()
					# ev_sent.synchronize()
					# tensorReadyTimes.append(ev_start.elapsed_time(ev_tensorReady))
					# sentTimes.append(ev_tensorReady.elapsed_time(ev_sent))

					# print("tneosrReadyTime: %.3f ms" % (sum(tensorReadyTimes)/ len(tensorReadyTimes)))
					# print("SentTime: %.3f ms" % (sum(sentTimes)/ len(sentTimes)))
					# if i % num_iters == 0:
					# print('Rank', rank, ' received data ', tensor)

def init_process(i, size, fn, backend='nccl'):
	rank = i
	with open(PRIVATE_ADDR_FILENAME, "r") as f:
		privateIps = []
		for line in f:
			privateIps.extend(line.split())
		coordinatorAddr = privateIps[0]
		print("[ClusterClient] auto filled coodinator address: ", coordinatorAddr)
	dist.init_process_group(backend, rank=rank, world_size=size,
							init_method='tcp://%s:29501'%coordinatorAddr) # set for master IP and free port for comm
	torch.cuda.set_device(rank)
	print('process ', rank, ' up!')
	fn(rank, size)

if __name__ == '__main__':
	size = 2
	processes = []
	print('torch.cuda availability: ', torch.cuda.is_available())
	print('torch.distributed availability: ', dist.is_available())
	print('how many GPUs?: ', torch.cuda.device_count())
	mp.spawn(init_process, args=(size, run), nprocs=size)
	# for rank in range(size):
	# 	p = mp.Process(target=init_process, args=(rank, size, run))
	# 	p.start()
	# 	processes.append(p)

	# for p in processes:
	# 	p.join()