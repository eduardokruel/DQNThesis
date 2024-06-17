#!/bin/sh
#SBATCH --time=04:00:00
#SBATCH -N 1

mkdir -p $TMPDIR/etl360
cp -r $HOME/slurm-bpai-tutorial/python $TMPDIR/etl360
cd $TMPDIR/etl360/python/
conda init
conda activate dqn39
python evaluate2.py --seed 3 --kill_reward 0 --save_path wizard_of_wor_v3__experiment2__3__1718536456
mkdir -p $HOME/slurm-bpai-tutorial/python/results
cp -r $TMPDIR/etl360/python/runs/ $HOME/slurm-bpai-tutorial/python/results
rm -rf $TMPDIR/etl360/python