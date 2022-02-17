#!/bin/bash

set -x
set -e

git submodule update --init --recursive .
pip3 install -r requirements.txt

# GRPC build for python
python -m grpc_tools.protoc -Icsrc/protos --python_out=. --grpc_python_out=. csrc/protos/runtime.proto

pushd bench
python setup.py install &
setuppid=$!
popd

# CPP runtime build.
mkdir -p csrc/build
pushd csrc/build/
cmake ..
make -j
popd

wait $setuppid
