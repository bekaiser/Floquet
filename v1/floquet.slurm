#!/bin/bash

# Job
#SBATCH --job-name=floq

# Partition             Nodes   S-C-T   Timelimit
# ---------             -----   -----   ---------
# sched_mit_hill        (32)    2-8-1   12:00:00
# sched_any_quicktest   2       2-8-1   00:15:00
# newnodes              (32)    2-10-1  12:00:00


#SBATCH --partition=sched_mit_raffaele
#SBATCH --nodes=1
#SBATCH --tasks-per-node=20
#SBATCH --time=48:00:00

# Streams
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

# Content
source ~/dedalus_paths
#module load engaging/anaconda/2.3.0 engaging/python/3.6.0 harvard/centos6/h5py-2.2.0_python-2.7.3
module list
python3 floquet.py
#mpirun -n 20 python3 qp.py


