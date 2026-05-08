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

# End-to-end polychromatic pipeline:
#   train → predict → scale (Careless) → refine (Phenix) → analysis (peaks + CC_anom)
#
# Usage:
#   sbatch train.sh <config> <surrogate> [wb_project]
#
# Scaling, refinement, and analysis are submitted as dependent SLURM jobs
# (different envs) and run after this job completes.
#
# Set INTEGRATOR_ROOT in your shell rc:
#     export INTEGRATOR_ROOT=/path/to/integrator

set -euo pipefail
export TQDM_DISABLE=1

# --- Args ---
config_arg=${1:?  "Usage: train.sh <config> <surrogate> [wb_project]"}
surrogate=${2:?   "Usage: train.sh <config> <surrogate> [wb_project]"}
wb_project=${3:-"PolyModel"}

# Careless configs to run (space-separated)
SCALE_CONFIGS="${SCALE_CONFIGS:-3 4 6}"

# --- Paths ---
INTEGRATOR_ROOT="${INTEGRATOR_ROOT:?INTEGRATOR_ROOT must be set (add to ~/.bashrc).}"
INTEGRATOR_CONFIGS="${INTEGRATOR_CONFIGS:-$INTEGRATOR_ROOT/configs}"
REFLTORCH_ROOT="${REFLTORCH_ROOT:?REFLTORCH_ROOT must be set (add to ~/.bashrc).}"
SCRIPTS="$REFLTORCH_ROOT/scripts/laue_output"

# --- Resolve config ---
if [[ -f "$config_arg" ]]; then
  config="$(realpath "$config_arg")"
elif [[ -f "$INTEGRATOR_CONFIGS/$config_arg" ]]; then
  config="$(realpath "$INTEGRATOR_CONFIGS/$config_arg")"
else
  echo "ERROR: config '$config_arg' not found." >&2
  exit 1
fi

echo "===== Resolved config: $config ====="

# --- Derive labels ---
read -r model_name loss_name profile_surr i_prior bg_prior < <(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('$config'))
model   = cfg['integrator']['name']
loss    = cfg['loss']['name']
profile = cfg['surrogates']['qp']['name']
pi  = (cfg['loss'].get('args',{}).get('pi_cfg',{})  or {}).get('name','none')
pbg = (cfg['loss'].get('args',{}).get('pbg_cfg',{}) or {}).get('name','none')
short = {'exponential':'expo', 'gamma':'gamma', 'half_cauchy':'hc',
         'log_normal':'ln', 'none':'none'}
print(model, loss, profile, short.get(pi,pi), short.get(pbg,pbg))
")
config_label="${model_name}_${loss_name}_${profile_surr}"
run_label="${config_label}_${surrogate}_pi-${i_prior}_pbg-${bg_prior}"

# --- Environment ---
source /n/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh
micromamba activate integrator

# --- Run directory ---
run_dir="${run_label}_${SLURM_JOB_ID}"
mkdir -p "$run_dir"

echo "===== Config:    $config ====="
echo "===== Surrogate: $surrogate ====="
echo "===== Run dir:   $(realpath "$run_dir") ====="

# =====================================================================
# 1. Train
# =====================================================================
echo "===== Starting integrator.train ====="
integrator.train -v \
  --config "$config" \
  --wb-project "$wb_project" \
  --save-dir "/n/netscratch/hekstra_lab/Lab/laldama/lightning_logs/" \
  --qbg "$surrogate" \
  --qi "$surrogate" \
  --run-dir "$run_dir" \
  --tags "$config_label" "$surrogate" "pi-${i_prior}" "pbg-${bg_prior}"

# =====================================================================
# 2. Predict + write MTZ
# =====================================================================
echo "===== Starting integrator.pred ====="
integrator.pred -v \
  --run-dir "$run_dir" \
  --write-mtz preds.mtz \
  --save-preds-as parquet

# =====================================================================
# 3. Submit Careless scaling (separate SLURM jobs — crls env)
# =====================================================================
echo "===== Submitting Careless scaling (configs: $SCALE_CONFIGS) ====="
scaling_output=$(python "$SCRIPTS/submit_scaling.py" \
  --run-dir "$run_dir" \
  --configs $SCALE_CONFIGS)
echo "$scaling_output"

# Extract colon-separated job IDs for dependency chaining
SCALING_IDS=$(echo "$scaling_output" | grep '^SCALING_JOB_IDS=' | cut -d= -f2)
if [[ -z "$SCALING_IDS" ]]; then
  echo "WARNING: No scaling job IDs captured; refinement/analysis will not be submitted."
  exit 0
fi

# =====================================================================
# 4. Submit refinement (depends on all scaling jobs)
# =====================================================================
echo "===== Submitting refinement (depends on scaling: $SCALING_IDS) ====="
refinement_output=$(python "$SCRIPTS/submit_refinement.py" \
  --run-dir "$run_dir" \
  --configs $SCALE_CONFIGS \
  --dependency "$SCALING_IDS")
echo "$refinement_output"

REFINEMENT_IDS=$(echo "$refinement_output" | grep '^REFINEMENT_JOB_IDS=' | cut -d= -f2)
if [[ -z "$REFINEMENT_IDS" ]]; then
  echo "WARNING: No refinement job IDs captured; analysis will not be submitted."
  exit 0
fi

# =====================================================================
# 5. Submit analysis (depends on all refinement jobs)
# =====================================================================
echo "===== Submitting analysis (depends on refinement: $REFINEMENT_IDS) ====="
python "$SCRIPTS/submit_analysis.py" \
  --run-dir "$run_dir" \
  --configs $SCALE_CONFIGS \
  --dependency "$REFINEMENT_IDS"

echo ""
echo "===== Full pipeline submitted ====="
echo "  scaling    → refinement → analysis (SLURM dependency chain)"
echo "  monitor: squeue -u \$USER"
