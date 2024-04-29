#!/bin/sh
#SBATCH --time=00:15:00
#SBATCH -N 1


cp -r $HOME/slurm-bpai-tutorial/python $TMPDIR/etl360
cd $TMPDIR/etl360/python/
pip install -r requirements/requirements.txt
python dqn.py
mkdir -p $HOME/slurm-bpai-tutorial/python/results
cp -a /runs/. $HOME/slurm-bpai-tutorial/python/results