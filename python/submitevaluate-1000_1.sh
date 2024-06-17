#!/bin/sh
#SBATCH --time=04:00:00
#SBATCH -N 1

mkdir -p $TMPDIR/etl360
cp -r $HOME/slurm-bpai-tutorial/python $TMPDIR/etl360
cd $TMPDIR/etl360/python/
conda init
conda activate dqn39
python evaluate2.py --seed 1 --kill_reward 1000 --negative_reward --save_path wizard_of_wor_v3__experiment2__1__1718536461
mkdir -p $HOME/slurm-bpai-tutorial/python/results
cp -r $TMPDIR/etl360/python/runs/ $HOME/slurm-bpai-tutorial/python/results
rm -rf $TMPDIR/etl360/python