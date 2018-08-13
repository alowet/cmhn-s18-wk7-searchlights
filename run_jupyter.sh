#!/bin/bash
#SBATCH --partition cmhn-s18
#SBATCH --nodes 1
#SBATCH -A cmhn-s18
#SBATCH --time 4:00:00
#SBATCH --mem-per-cpu 12G
#SBATCH --job-name tunnel
#SBATCH --output jupyter-log-%J.txt

## get tunneling info
XDG_RUNTIME_DIR=""
ipnport=$(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)

## print tunneling instructions to jupyter-log-{jobid}.txt
echo -e "
    Copy/Paste this in your local terminal to ssh tunnel with remote
    -----------------------------------------------------------------
    ssh -N -L $ipnport:$ipnip:$ipnport $USER@milgram.hpc.yale.edu
    -----------------------------------------------------------------

    Then open a browser on your local machine to the following address
    ------------------------------------------------------------------
    localhost:$ipnport  (prefix w/ https:// if using password)
    ------------------------------------------------------------------
    "

## start an ipcluster instance and launch jupyter server
module load Langs/Python/3.5-anaconda
#module load Pypkgs/brainiak/0.5-anaconda
module load Pypkgs/brainiak/0.7.1-anaconda
module load Pypkgs/NILEARN/0.4.0-anaconda

module load MPI/OpenMPI
mpirun -n 1 jupyter-notebook --no-browser --port=$ipnport --ip=$ipnip
