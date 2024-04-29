#!/bin/sh
#SBATCH --time=00:15:00
#SBATCH -N 1

mkdir -p $TMPDIR/etl360
cp -r $HOME/slurm-bpai-tutorial/python $TMPDIR/etl360
cd $TMPDIR/etl360/python/
conda init
conda activate dqn39
python dqn.py --seed 1 --env-id CartPole-v1 --total-timesteps 500000 --track
mkdir -p $HOME/slurm-bpai-tutorial/python/results
cp -a /runs/. $HOME/slurm-bpai-tutorial/python/results
rm -rf $TMPDIR/etl360/python