# DeepPool Artifact

## Instructions on how to run the VGG example

Ensure you have NVIDIA docker available on your system
Download and run the PyTorch container:
```
docker run --gpus all --network="host" -it --rm nvcr.io/nvidia/pytorch:22.01-py3
```
In the container, clone the DeepPool repo:
```
git clone https://github.com/joshuafried/DeepPool-Artifact
````

Enter the directory and build DeepPool:
```
cd DeepPool-Artifact
bash build.sh
```

Now you can launch the DeepPool cluster coordinator as a background job:
```
python3 cluster.py  --addrToBind 0.0.0.0:12347 --c10dBackend nccl --be_batch_size=0 --cpp --logdir=$PWD &
```

Once you see "Now, cluster is ready to accept training jobs." you may launch a job.
For example, to run VGG across 8 GPUs in DataParallel mode run:
```
python3 examples/vgg.py 8 32 DP
```
To run VGG in BurstParallel mode:
```
python3 examples/vgg.py 8 32 5.0
```

To view the results of the run, inspect the contents of cpprt0.out:
```
tail cpprt0.out
```
When a job completes, you will see a line of output indicating the iteration such as:
```
A training job vgg16_8_32_2.0_DP is completed (1800 iters, 13.57 ms/iter, 73.71 iter/s, 0.00 be img/s, 32 globalBatchSize).
```


To kill the cluster, run
```
pkill runtime
```

Now re-run VGG with a background training job:
```
python3 examples/vgg_be.py
python3 cluster.py  --addrToBind 0.0.0.0:12347 --c10dBackend nccl --be_batch_size=8  --cpp --logdir=$PWD --be_jit_file=vgg.jit &
```
Once the cluster is running:
```
python3 examples/vgg.py 8 32 DP 1
python3 examples/vgg.py 8 32 5.0 1
```
