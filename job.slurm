#!/bin/bash
#SBATCH --job-name=delta_stepping
#SBATCH --output=delta_stepping_%j.out
#SBATCH --error=delta_stepping_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=rome

module load python3/3.11.4
module load mpi/openmpi/4.1.4

cd ~/delta_stepping/

python3 precompute_scores.py
ccc_mprun -n 8 python3 delta_stepping_mpi.py
python3 visualize_chemin.py
