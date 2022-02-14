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

