#!/bin/bash

dataset=$1
model=$2

CUBLAS_WORKSPACE_CONFIG=:4096:8 python run_${model}.py --dataset ${dataset} > ${model}_${dataset}.out 2>&1

if [ "$model" = "bprmf" ]; then
    CUBLAS_WORKSPACE_CONFIG=:4096:8 python run_multimodal.py --dataset ${dataset} --model vbpr --method zeros > vbpr_${dataset}_zeros.out 2>&1
    CUBLAS_WORKSPACE_CONFIG=:4096:8 python run_multimodal.py --dataset ${dataset} --model vbpr --method random > vbpr_${dataset}_random.out 2>&1
    CUBLAS_WORKSPACE_CONFIG=:4096:8 python run_multimodal.py --dataset ${dataset} --model vbpr --method mean > vbpr_${dataset}_mean.out 2>&1
else
    CUBLAS_WORKSPACE_CONFIG=:4096:8 python run_multimodal.py --dataset ${dataset} --model ${model}m --method zeros > ${model}m_${dataset}_zeros.out 2>&1
    CUBLAS_WORKSPACE_CONFIG=:4096:8 python run_multimodal.py --dataset ${dataset} --model ${model}m --method random > ${model}m_${dataset}_random.out 2>&1
    CUBLAS_WORKSPACE_CONFIG=:4096:8 python run_multimodal.py --dataset ${dataset} --model ${model}m --method mean > ${model}m_${dataset}_mean.out 2>&1
fi