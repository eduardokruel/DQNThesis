#!/bin/sh
#SBATCH --time=00:15:00
#SBATCH -N 1
-c 24

module load python/
cp -r $HOME/run3 $TMPDIR
cd $TMPDIR/run3
python test.py
mkdir -p $HOME/run3/results
cp result.dat run3.log $HOME/run3/results