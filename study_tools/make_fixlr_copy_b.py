#!/usr/bin/env python3
"""Make a separate copy of the code with the pretraining learning-rate schedule fixed, for a check run.
The main folder (/data/tchitando/Non_neg) is only read: nothing in it is changed, so the runs training from it are
unaffected and every new run they start still uses the original code.

    python make_fixlr_copy_b.py                      # makes /data/tchitando/Non_neg_fixlr
    python make_fixlr_copy_b.py --dry-run            # shows what it would do

What it does:
  1. copies solo/, MedMNIST/ and study_tools/ to the new folder (no checkpoints, caches or data); the data files
     (*.npz) are linked, not copied;
  2. fixes the two places that make the schedule repeat:
       solo/methods/base.py      without a datamodule (always, in main_pretrain.py) steps_per_epoch = max_epochs = 100;
                                 the Mac copy's version uses len(_dl) instead. Either is replaced by
                              -> steps_per_epoch = trainer.estimated_stepping_batches / max_epochs   (as in NCL/BYI)
       solo/utils/lr_scheduler.py  progress is capped at 1, so the cosine stops at its end instead of restarting
  3. checks the fixed scheduler on a dummy optimiser: 10 warmup epochs, one cosine to the minimum at epoch 100;
  4. writes run_fixlr.sh in the new folder: SimCLR, seed 42, the SimCLR defaults first, then brightness + contrast,
     brightness 0.8 and crop only, each named <run id>-fixlr. Checkpoints and metrics go to the new folder only.
"""
import argparse, importlib.util, math, re, shutil, sys
from pathlib import Path

SRC = Path("/data/tchitando/Non_neg")
DST = Path("/data/tchitando/Non_neg_fixlr")
CFGS = ["simclr_rn50_p_cifar", "simclr_rn50_p_croponly_bright_contrast_f030", "simclr_rn50_p_croponly_bright",
        "simclr_rn50_p_croponly"]
OCT = "MedMNIST/scripts/pretrain/octmnist"


def patch_once(path, pattern, repl, what, dry):
    text = path.read_text()
    new, n = re.subn(pattern, repl, text)
    if n != 1:
        raise SystemExit(f"{path}: expected exactly one '{what}', found {n}; nothing patched")
    compile(new, str(path), "exec")
    if not dry:
        path.write_text(new)
    print(f"fixed   {path.relative_to(path.parents[2]) if len(path.parents) > 2 else path}: {what}")


FIX = "steps_per_epoch = self.trainer.estimated_stepping_batches / self.max_epochs  # fixlr: real optimiser steps per epoch"
# the server's version (25 May): real count only with a datamodule, otherwise max_epochs (= 100 steps per epoch)
SERVER = (r"(?m)^(?P<ind>[ \t]*)if hasattr\(self\.trainer, 'datamodule'\) and self\.trainer\.datamodule is not None:[^\n]*\n"
          r"[ \t]*steps_per_epoch = len\(self\.trainer\.datamodule\.train_dataloader\(\)\)[^\n]*\n"
          r"[ \t]*else:[ \t]*\n"
          r"[ \t]*steps_per_epoch = self\.max_epochs[ \t]*\n")
# the Mac copy's version (6 Oct)
MAC = r"(?m)^(?P<ind>[ \t]*)steps_per_epoch\s*=\s*len\(_dl\)\s*//\s*self\.trainer\.accumulate_grad_batches[^\n]*\n"


def patch_base(path, dry):
    text = path.read_text()
    hits = [(name, pat) for name, pat in (("server version (datamodule, else max_epochs)", SERVER), ("Mac version (len(_dl))", MAC))
            if len(re.findall(pat, text)) == 1]
    if len(hits) != 1:
        raise SystemExit(f"{path}: could not identify the steps_per_epoch code (matched: {[h[0] for h in hits]}); nothing patched")
    name, pat = hits[0]
    new = re.sub(pat, lambda m: m.group("ind") + FIX + "\n", text, count=1)
    compile(new, str(path), "exec")
    if not dry:
        path.write_text(new)
    print(f"fixed   solo/methods/base.py ({name}): steps_per_epoch from estimated_stepping_batches")


def check_scheduler(path):
    import torch
    spec = importlib.util.spec_from_file_location("lr_fixed", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    steps, epochs, warm, base = 380, 100, 10, 0.4
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=base)
    sch = mod.LinearWarmupCosineAnnealingLR(opt, warmup_epochs=warm * steps, max_epochs=steps * epochs,
                                            warmup_start_lr=0.0, eta_min=0.0)
    lr_end = {}
    for t in range(1, steps * epochs + 2 * steps + 1):          # 2 epochs past the end, to see that it stays down
        opt.step(); sch.step()
        if t % steps == 0:
            lr_end[t // steps - 1] = opt.param_groups[0]["lr"]
    show = [0, 1, 9, 10, 49, 98, 99, 100, 101]
    print("check   learning rate at the end of epoch: " + ", ".join(f"{e}: {lr_end[e]:.4f}" for e in show))
    ok = (abs(lr_end[9] - base) < 1e-6 and abs(lr_end[49] - base * 0.5 * (1 + math.cos(math.pi * 40 / 90))) < 1e-3
          and lr_end[99] < 1e-6 and lr_end[100] < 1e-6 and lr_end[101] < 1e-6)
    if not ok:
        raise SystemExit("the fixed scheduler does not give warmup to epoch 10 then one cosine to 0 at epoch 100")
    print("check   ok: warmup to 0.4 by epoch 10, one cosine to 0 at epoch 100, stays at 0 afterwards")


LAUNCH = """#!/usr/bin/env bash
# Learning-rate check (separate copy, schedule fixed): SimCLR, seed 42, one run after another.
#   CUDA_VISIBLE_DEVICES=1 bash run_fixlr.sh
cd "$(dirname "$0")" || exit 1
for CFG in @CFGS@; do
  NAME=$(sed -nE 's/^name:[[:space:]]*"?([^"]*)"?[[:space:]]*$/\\1/p' @OCT@/$CFG.yaml)-fixlr
  if find trained_models -name "$NAME-offline-*-ep=99.ckpt" 2>/dev/null | grep -q .; then echo "done already: $NAME"; continue; fi
  echo "=== $NAME (GPU ${CUDA_VISIBLE_DEVICES:-?})"
  python3 MedMNIST/main_pretrain.py --config-path @OCT@ --config-name "$CFG" name="$NAME" auto_resume.enabled=False
done
echo "fixlr runs finished"
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if DST.exists():
        raise SystemExit(f"{DST} already exists; not touching it")
    missing = [c for c in CFGS if not (SRC / OCT / f"{c}.yaml").exists()]
    if missing:
        raise SystemExit(f"run files not found in {SRC / OCT}: {missing}")
    ign = shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints", "*.npz", "*.ckpt", "*.pyc")
    print(f"copy    {SRC} -> {DST}: solo/, MedMNIST/, study_tools/ (no data, checkpoints or caches)")
    npz = sorted((SRC / "MedMNIST").rglob("*.npz"))
    print(f"link    {len(npz)} data files: " + ", ".join(str(p.relative_to(SRC)) for p in npz))
    if a.dry_run:
        tmp = None
    else:
        DST.mkdir()
        for d in ("solo", "MedMNIST", "study_tools"):
            shutil.copytree(SRC / d, DST / d, ignore=ign)        # real copies: editing them cannot touch the originals
        for p in npz:
            q = DST / p.relative_to(SRC); q.parent.mkdir(parents=True, exist_ok=True); q.symlink_to(p)
        (DST / "trained_models").mkdir(); (DST / "study").mkdir()
    root = SRC if a.dry_run else DST
    patch_base(root / "solo/methods/base.py", dry=a.dry_run)
    patch_once(root / "solo/utils/lr_scheduler.py",
               r"progress\s*=\s*\(self\.last_epoch\s*-\s*self\.warmup_epochs\)\s*/\s*max\(1,\s*self\.max_epochs\s*-\s*self\.warmup_epochs\)",
               "progress = min(1.0, (self.last_epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs))  # fixlr: no restart",
               "progress capped at 1", dry=True if a.dry_run else False)
    if a.dry_run:
        print("dry run: nothing copied or changed"); return
    check_scheduler(DST / "solo/utils/lr_scheduler.py")
    sh = DST / "run_fixlr.sh"
    sh.write_text(LAUNCH.replace("@CFGS@", " ".join(CFGS)).replace("@OCT@", OCT)); sh.chmod(0o755)
    print(f"wrote   {sh}  (runs, in order: " + ", ".join(CFGS) + "; all seed 42, names end in -fixlr)")
    print(f"\nnext:   cd {DST} && CUDA_VISIBLE_DEVICES=1 bash run_fixlr.sh   (in tmux)")


if __name__ == "__main__":
    main()
