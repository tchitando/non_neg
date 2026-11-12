#!/usr/bin/env bash
# Run this on the RunPod pod AFTER uploading the Non_neg code via rsync.
# Usage: bash /workspace/Non_neg/study_tools/setup_runpod.sh
set -e
cd /workspace/Non_neg

echo "=== Installing dependencies ==="
pip install --quiet pytorch-lightning==1.6.4 torchmetrics==0.6.0 einops lightning-bolts tqdm wandb scipy timm scikit-learn hydra-core medmnist

echo "=== Downloading OCT 64x64 data ==="
python3 -c "
import medmnist
ds = medmnist.OCTMnist(split='train', download=True, size=64, root='/workspace/Non_neg/MedMNIST')
print('octmnist_64.npz downloaded to MedMNIST/')
"

echo "=== Applying fixlr patches ==="
python3 - <<'PYEOF'
import re, sys

def patch(path, pattern, repl, label):
    text = open(path).read()
    new, n = re.subn(pattern, repl, text)
    if n != 1:
        sys.exit(f"ERROR: expected 1 match for '{label}' in {path}, got {n}")
    open(path, 'w').write(new)
    print(f"patched {path}: {label}")

patch('solo/methods/base.py',
      r'steps_per_epoch\s*=\s*len\(_dl\)\s*//\s*self\.trainer\.accumulate_grad_batches',
      'steps_per_epoch = self.trainer.estimated_stepping_batches / self.max_epochs  # fixlr',
      'steps_per_epoch from estimated_stepping_batches')

patch('solo/utils/lr_scheduler.py',
      r'progress\s*=\s*\(self\.last_epoch\s*-\s*self\.warmup_epochs\)\s*/\s*max\(1,\s*self\.max_epochs\s*-\s*self\.warmup_epochs\)',
      'progress = min(1.0, (self.last_epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs))  # fixlr',
      'progress capped at 1')
PYEOF

echo "=== Setup complete — ready to train ==="
echo "Run: bash /workspace/Non_neg/study_tools/run_fixlr_pod.sh"
