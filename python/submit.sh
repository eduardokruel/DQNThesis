#!/bin/sh
#SBATCH --time=00:15:00
#SBATCH -N 1
-c 24


cp -r $HOME/slurm-bpai-tutorial/python $TMPDIR
cd $TMPDIR/python
python script.py
mkdir -p $HOME/slurm-bpai-tutorial/python/results
cp result.dat run3.log $HOME/slurm-bpai-tutorial/python/results