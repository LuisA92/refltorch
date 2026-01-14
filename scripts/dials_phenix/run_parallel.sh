#!/bin/bash
#SBATCH -c 16
#SBATCH -N 1
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1 
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200G
#SBATCH -t 0-10:00
#SBATCH -o pytorch_%j.out
#SBATCH -e pytorch_%j.err

# hyperparameters
# scripts for dials/phenix
refltorch_dir=/n/hekstra_lab/people/aldama/refltorch/scripts/dials_phenix

# source micromamba
source /n/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh

# activate micromamba environment
micromamba activate integrator

# make run directory
run_dir=run_$SLURM_JOB_ID
mkdir $run_dir

# make a copy of the config.yaml file
cp intgratr.yaml run_$SLURM_JOB_ID/config_copy.yaml

echo "===== Starting integrator.train ====="
# train integrator
integrator.train -v \
    --config intgratr.yaml \
    --wb-project GammRepam \
    --run-dir $run_dir

echo "===== Starting integrator.pred ====="
# get predictions
integrator.pred -v \
  --run-dir $run_dir \
  --write-refl

echo "===== Starting DIALS-Phenix Parallel Processing Setup ====="

# activate refltorch environment
micromamba deactivate
micromamba activate refltorch

# create logs directory
mkdir -p logs

# Generate config files
python $refltorch_dir/create_config.py --run-dir $run_dir

# Submit jobs for scaling/merging/find_peaks
python $refltorch_dir/submit_jobs.py $run_dir
