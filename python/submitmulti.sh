#!/bin/sh
#SBATCH --time=00:15:00
#SBATCH -N 1

mkdir -p $TMPDIR/etl360
cp -r $HOME/slurm-bpai-tutorial/python $TMPDIR/etl360
cd $TMPDIR/etl360/python/
conda init
conda activate dqn39
python ppo_pettingzoo_ma_atari.py --seed 1 --env-id wizard_of_wor_v3 --total-timesteps 50000 --track --wandb-project-name cleanrldasmulti --capture_video
mkdir -p $HOME/slurm-bpai-tutorial/python/results
cp -a /runs/. $HOME/slurm-bpai-tutorial/python/results
rm -rf $TMPDIR/etl360/python