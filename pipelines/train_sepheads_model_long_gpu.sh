#!/bin/bash

#SBATCH --job-name="phb_sepheads_model_training"
#SBATCH --account=moscato
#SBATCH --partition=long_gpu
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --mem=64000MB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emanuele.moscato2@unibocconi.it
#SBATCH --output=../slurm_out/%x_%j.out
# #SBATCH --error=../slurm_err/%x_%j.err

module load miniconda3
source /home/Moscato/.bashrc

conda activate phb

cd ../scripts/
echo "Switched to directory: $(pwd)"

echo "Launching Python script"

# Note: -u = unbuffered stdout and stderr (so they get propagated immediately).
python -u train_sepheads_model.py
    
echo "Python script ended"
