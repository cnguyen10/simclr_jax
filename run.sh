#!/bin/bash
DEVICE_ID=0  # which GPU is going to be used
export CUDA_VISIBLE_DEVICES=${DEVICE_ID}

python3 "main.py"