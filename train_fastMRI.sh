#!/bin/bash

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

if [ "$1" = "vp" ]
then
    echo "================ run configs/vp/ddpm_continuous.py ================"
    python main.py \
        --config=configs/vp/ddpm_continuous.py \
        --mode='train' \
        --workdir=results \
        ${2:+--resume_dir="$2"}

elif [ "$1" = "ve" ]
then
    echo "================ run configs/ve/ncsnpp_continuous.py ================"
    python main.py \
        --config=configs/ve/ncsnpp_continuous.py \
        --mode='train' \
        --workdir=results \
        ${2:+--resume_dir="$2"}
else
    echo "================ You must input one argument: ve or vp ================"
    echo "Usage: bash train_fastMRI.sh <ve|vp> [resume_dir]"
fi