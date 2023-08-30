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

export LLM_MODEL_TYPE=openai

export OPENAI_MODEL_NAME="gpt-3.5-turbo"
echo Testing $OPENAI_MODEL_NAME
python test.py 2>&1 | tee ./data/logs/openai-${OPENAI_MODEL_NAME}_${EXT}.log

export OPENAI_MODEL_NAME="gpt-4"
echo Testing $OPENAI_MODEL_NAME
python test.py 2>&1 | tee ./data/logs/openai-${OPENAI_MODEL_NAME}_${EXT}.log
