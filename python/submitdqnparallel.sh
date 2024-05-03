#!/bin/sh
#SBATCH --time=00:15:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=2

mkdir -p $TMPDIR/etl360
cp -r $HOME/slurm-bpai-tutorial/python $TMPDIR/etl360
cd $TMPDIR/etl360/python/
conda init
conda activate dqn39
python dqn.py --seed $SLURM_ARRAY_TASK_ID --env-id CartPole-v1 --total-timesteps 500000 --track --wandb-project-name cleanrldas --capture_video
ls
mkdir -p $HOME/slurm-bpai-tutorial/results
cp -r $TMPDIR/etl360/python/wandb $HOME/slurm-bpai-tutorial/results
rm -rf $TMPDIR/etl360/python