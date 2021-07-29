#!/bin/bash
set -x

# Packages to install
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
conda install grpcio
conda install grpcio-tools
