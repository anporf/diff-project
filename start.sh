#!/bin/sh

wget -O cond-vp.pkl https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
git clone https://github.com/NVlabs/edm
cp -r edm/dnnlib dnnlib
cp -t edm/torch_utils torch_utils