#!/bin/sh
#SBATCH --time=00:15:00
#SBATCH -N 1

mkdir -p $TMPDIR/etl360
cp -r $HOME/slurm-bpai-tutorial/python $TMPDIR/etl360
cd $TMPDIR/etl360/python/
conda init
conda activate dqn39
python evaluate.py --seed 1 --kill_reward 0 --save_model wizard_of_wor_v3__experiment__1__1717437083
mkdir -p $HOME/slurm-bpai-tutorial/python/results
cp -r $TMPDIR/etl360/python/runs/ $HOME/slurm-bpai-tutorial/python/results
rm -rf $TMPDIR/etl360/python