#!/usr/bin/env bash
# Run the fixlr check on RunPod: SimCLR + NCL, both with CIFAR-like (p_cifar) augmentations.
# Usage: bash /workspace/Non_neg/study_tools/run_fixlr_pod.sh
set -e
cd /workspace/Non_neg
export PYTHONPATH=/workspace/Non_neg:$PYTHONPATH
mkdir -p trained_models

OCT="MedMNIST/scripts/pretrain/octmnist"
for CFG in simclr_rn50_p_cifar ncl_rn50_p_cifar; do
    NAME=$(sed -nE 's/^name:[[:space:]]*"?([^"]*)"?[[:space:]]*$/\1/p' $OCT/$CFG.yaml)-fixlr
    echo "=== $NAME ==="
    python3 MedMNIST/main_pretrain.py \
        --config-path $OCT \
        --config-name "$CFG" \
        name="$NAME" \
        auto_resume.enabled=False \
        checkpoint.keep_prev=False
done
echo "fixlr runs finished"
