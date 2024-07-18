#!/bin/bash

#SBATCH --job-name=run-python
#SBATCH --mail-type=ALL
#SBATCH --mail-user=enuss@uw.edu

#SBATCH --account=nearshore
#SBATCH --partition=compute
# #SBATCH --partition=ckpt-all
#SBATCH -c 1
#SBATCH --time=72:00:00
#SBATCH --mem=100G

#SBATCH --export=all
#SBATCH --output=run-python.out # where STDOUT goes
#SBATCH --error=run-python.err # where STDERR goes

# Modules to use (optional).

# Your programs to run.
python alongwavenum-figures.py
