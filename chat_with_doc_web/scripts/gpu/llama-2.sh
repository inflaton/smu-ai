#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/../../../chat-with-pci-dss-v4/
echo Current Directory:
pwd

nvidia-smi
uname -a
cat /etc/os-release
lscpu
grep MemTotal /proc/meminfo

export TRANSFORMERS_CACHE=/common/scratch/users/d/dh.huang.2023/transformers

export EXT=new_cluster_a40

export HUGGINGFACE_MODEL_NAME_OR_PATH="meta-llama/Llama-2-7b-chat-hf"
echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
python test.py 2>&1 | tee ./data/logs/Llama-2-7b-chat-hf_${EXT}.log

export HUGGINGFACE_MODEL_NAME_OR_PATH="meta-llama/Llama-2-13b-chat-hf"
echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
python test.py 2>&1 | tee ./data/logs/Llama-2-13b-chat-hf_${EXT}.log

export EXT=new_cluster_a40_8bit
export LOAD_QUANTIZED_MODEL=8bit

# export HUGGINGFACE_MODEL_NAME_OR_PATH="meta-llama/Llama-2-7b-chat-hf"
# echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
# python test.py 2>&1 | tee ./data/logs/Llama-2-7b-chat-hf_${EXT}.log

# export HUGGINGFACE_MODEL_NAME_OR_PATH="meta-llama/Llama-2-13b-chat-hf"
# echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
# python test.py 2>&1 | tee ./data/logs/Llama-2-13b-chat-hf_${EXT}.log

export EXT=new_cluster_a40_4bit
export LOAD_QUANTIZED_MODEL=4bit

# export HUGGINGFACE_MODEL_NAME_OR_PATH="meta-llama/Llama-2-7b-chat-hf"
# echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
# python test.py 2>&1 | tee ./data/logs/Llama-2-7b-chat-hf_${EXT}.log

# export HUGGINGFACE_MODEL_NAME_OR_PATH="meta-llama/Llama-2-13b-chat-hf"
# echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
# python test.py 2>&1 | tee ./data/logs/Llama-2-13b-chat-hf_${EXT}.log

export HUGGINGFACE_MODEL_NAME_OR_PATH="meta-llama/Llama-2-70b-chat-hf"
echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
python test.py 2>&1 | tee ./data/logs/Llama-2-70b-chat-hf_${EXT}.log
