#!/usr/bin/env bash
# Input python command to be submitted as a job

#SBATCH --output=searchlight-%j.out
#SBATCH --job-name searchlight
#SBATCH -p chmn-s18
#SBATCH -A chmn-s18
#SBATCH -t 30
#SBATCH -m=4G
#SBATCH -n 2

# Set up the environment
module load Langs/Python/3.5-anaconda
module load Pypkgs/brainiak/0.5-anaconda
module load Pypkgs/NILEARN/0.4.0-anaconda
module load MPI/OpenMPI

# Run the python script
srun -n $SLURM_NTASKS --mpi=pmi2 python searchlight.py
