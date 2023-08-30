#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/../learn-ai
echo Current Directory:
pwd

export TRANSFORMERS_CACHE=/common/scratch/users/d/dh.huang.2023/transformers
export PORT=64300

make serve
