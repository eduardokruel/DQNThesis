#!/bin/sh
#SBATCH --time=00:15:00
#SBATCH -N 1
-c 24

module load python/
cp -r $HOME/python $TMPDIR
cd $TMPDIR/python
python test.py
mkdir -p $HOME/python/results
cp result.dat run3.log $HOME/run3/results