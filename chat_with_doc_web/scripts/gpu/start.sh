#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

sbatch scripts/sbatch.sh scripts/serve.sh
