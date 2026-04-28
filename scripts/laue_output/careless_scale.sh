#!/usr/bin/env bash
#SBATCH --job-name=careless
#SBATCH -p gpu
#SBATCH --mem=100G
#SBATCH -t 0-10:00
#SBATCH --gres=gpu:1
#SBATCH -o careless_%j.out
#SBATCH -e careless_%j.err

# Unified Careless scaling script — replaces config1..6.sh
#
# Usage:
#   sbatch careless_scale.sh <epoch_dir> <config> [dmin]
#
# <epoch_dir>  Directory containing preds.mtz (output of integrator.pred)
# <config>     1-6 (see table below)
# [dmin]       Resolution cutoff in Å (default: 1.5)
#
# Configs:
#   1  single mtz, anomalous, baseline
#   2  friedel split, separate-files, double-wilson
#   3  friedel split, separate-files, double-wilson, pos-encoding
#   4  single mtz, anomalous, pos-encoding
#   5  single mtz, anomalous, image-layers=2, mlp-width=24, 6k iters
#   6  single mtz, anomalous, image-layers=2, mlp-width=24, pos-encoding

set -euo pipefail

epoch_dir="${1:?Usage: careless_scale.sh <epoch_dir> <config> [dmin]}"
config="${2:?Usage: careless_scale.sh <epoch_dir> <config> [dmin]}"
dmin="${3:-1.5}"

source /n/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh
micromamba activate crls

scaling_dir="$epoch_dir/scaling"
mkdir -p "$scaling_dir"
outprefix="$scaling_dir/config${config}"

# Shared flags
common=(
    --merge-half-datasets
    --half-dataset-repeats=3
    --mc-samples=10
    --mlp-layers=10
    --dmin="$dmin"
    --studentt-likelihood-dof=64
    --wavelength-key="wavelength"
)

pos_encoding=(
    --positional-encoding-frequencies=4
    --positional-encoding-keys="xcal,ycal,BATCH"
)

# Friedel splitting for configs 2,3
_friedel_split() {
    local mtz="$1" prefix="$2"
    python3 -c "
import reciprocalspaceship as rs
ds = rs.read_mtz('$mtz')
mask = ds['SIGI'] > 0
ds = ds[mask]
plus = ds.hkl_to_asu()['M/ISYM'].to_numpy() % 2 == 1
centrics = ds.label_centrics().CENTRIC.to_numpy()
plus |= centrics
ds[plus].write_mtz('${prefix}_friedel_plus.mtz')
ds[~plus].write_mtz('${prefix}_friedel_minus.mtz')
print(f'split {len(ds)} refls: {plus.sum()} plus, {(~plus).sum()} minus')
"
}

case "$config" in
    1)
        careless poly \
            --anomalous \
            "${common[@]}" \
            --mlp-width=32 \
            --image-layers=0 \
            --iterations=30000 \
            "BATCH,xcal,ycal,dHKL,wavelength" \
            "$epoch_dir/preds.mtz" \
            "$outprefix"
        ;;
    2)
        _friedel_split "$epoch_dir/preds.mtz" "$outprefix"
        careless poly \
            --separate-files \
            "${common[@]}" \
            --mlp-width=32 \
            --image-layers=0 \
            --test-fraction=0.1 \
            --iterations=30000 \
            --double-wilson-parents=None,0 \
            --double-wilson-r=0.,0.99902 \
            --seed=$RANDOM \
            "BATCH,xcal,ycal,dHKL,wavelength" \
            "${outprefix}_friedel_plus.mtz" \
            "${outprefix}_friedel_minus.mtz" \
            "$outprefix"
        ;;
    3)
        _friedel_split "$epoch_dir/preds.mtz" "$outprefix"
        careless poly \
            --separate-files \
            "${common[@]}" \
            --mlp-width=32 \
            --image-layers=0 \
            "${pos_encoding[@]}" \
            --test-fraction=0.1 \
            --iterations=30000 \
            --double-wilson-parents=None,0 \
            --double-wilson-r=0.,0.99902 \
            --seed=$RANDOM \
            "BATCH,xcal,ycal,dHKL,wavelength" \
            "${outprefix}_friedel_plus.mtz" \
            "${outprefix}_friedel_minus.mtz" \
            "$outprefix"
        ;;
    4)
        careless poly \
            --anomalous \
            "${common[@]}" \
            --mlp-width=32 \
            --image-layers=0 \
            "${pos_encoding[@]}" \
            --test-fraction=0.1 \
            --iterations=30000 \
            --seed=$RANDOM \
            "BATCH,xcal,ycal,dHKL,wavelength" \
            "$epoch_dir/preds.mtz" \
            "$outprefix"
        ;;
    5)
        careless poly \
            --anomalous \
            "${common[@]}" \
            --mlp-width=24 \
            --image-layers=2 \
            --test-fraction=0.1 \
            --iterations=6000 \
            --seed=$RANDOM \
            "BATCH,xcal,ycal,dHKL,wavelength" \
            "$epoch_dir/preds.mtz" \
            "$outprefix"
        ;;
    6)
        careless poly \
            --anomalous \
            "${common[@]}" \
            --mlp-width=24 \
            --image-layers=2 \
            "${pos_encoding[@]}" \
            --test-fraction=0.1 \
            --iterations=30000 \
            --seed=$RANDOM \
            "BATCH,xcal,ycal,dHKL,wavelength" \
            "$epoch_dir/preds.mtz" \
            "$outprefix"
        ;;
    *)
        echo "ERROR: unknown config '$config' (expected 1-6)" >&2
        exit 1
        ;;
esac

echo "===== Careless config${config} complete: $outprefix ====="
