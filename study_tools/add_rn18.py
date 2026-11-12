#!/usr/bin/env python3
"""Adds ResNet-18 checkpoints to the scoring scripts and writes the ResNet-18 OCT manifest. Run once, from the repo root:

    python study_tools/add_rn18.py

1. study_tools/eval_run.py: load_model reads the backbone (ResNet-18 or ResNet-50) and the projector layout from the
   checkpoint instead of assuming ResNet-50. ResNet-50 checkpoints load exactly as before. eval_spec.py uses the same
   function, so it needs no change. A backup is kept as eval_run.py.bak_before_rn18.
2. study/rn18/manifest.csv: the corrected ResNet-18 OCT runs, A0 (= the ResNet-50 stem) and A6 (= speckle + contrast,
   crop 0.2), seeds 42, 43, 44, both objectives. Seed 42 is the existing corrected run.

Scoring afterwards (results go to study/rn18/, the ResNet-50 tables are not touched):
    CUDA_VISIBLE_DEVICES=1 python study_tools/eval_run.py --manifest study/rn18/manifest.csv --master study/rn18/results_master.csv
    CUDA_VISIBLE_DEVICES=1 python study_tools/eval_spec.py --master study/rn18/results_master.csv --out study/rn18/results_spec.csv
"""
import csv, json, shutil, sys
from pathlib import Path

ER = Path("study_tools/eval_run.py")
MARK = "# rn18-support"

OLD = '''def load_model(ckpt_path):
    backbone = models.resnet50(weights=None)
    backbone.fc = nn.Identity()
    projector = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 256))
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck["state_dict"]
    backbone.load_state_dict({k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}, strict=True)
    projector.load_state_dict({k[len("projector."):]: v for k, v in sd.items() if k.startswith("projector.")}, strict=True)
    return backbone.cuda().eval(), projector.cuda().eval(), ck.get("epoch", "")
'''

NEW = '''def projector_from_state(psd):  # rn18-support
    """Rebuild the projector from its state-dict keys: an index with a 2-D weight is a Linear layer, one with
    running statistics is BatchNorm1d, and an index without parameters is a ReLU."""
    n = max(int(k.split(".")[0]) for k in psd) + 1
    layers = []
    for i in range(n):
        w = psd.get(f"{i}.weight")
        if w is None:
            layers.append(nn.ReLU())
        elif f"{i}.running_mean" in psd:
            layers.append(nn.BatchNorm1d(w.shape[0]))
        else:
            layers.append(nn.Linear(w.shape[1], w.shape[0], bias=f"{i}.bias" in psd))
    return nn.Sequential(*layers)


def load_model(ckpt_path):  # rn18-support: ResNet-18 or ResNet-50, read from the checkpoint
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck["state_dict"]
    bsd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    psd = {k[len("projector."):]: v for k, v in sd.items() if k.startswith("projector.")}
    backbone = (models.resnet50 if "layer1.0.conv3.weight" in bsd else models.resnet18)(weights=None)
    backbone.fc = nn.Identity()
    projector = projector_from_state(psd)
    backbone.load_state_dict(bsd, strict=True)
    projector.load_state_dict(psd, strict=True)
    return backbone.cuda().eval(), projector.cuda().eval(), ck.get("epoch", "")
'''

FIELDS = ["run_id", "part", "dataset", "backbone", "config", "objective", "seed", "policy_id", "bitmask", "augs_on",
          "n_augs", "prob", "levels", "crop_min_scale"]
CONFIGS = {
    "A0": dict(policy_id="stem", bitmask="000000", augs_on="", n_augs=0, levels="", crop_min_scale=0.08),
    "A6": dict(policy_id="speckle_contrast_crop020", bitmask="110000", augs_on="speckle+contrast", n_augs=2,
               levels=json.dumps({"speckle": {"std": 0.1}, "contrast": {"factor": 0.8}}), crop_min_scale=0.2),
}


def main():
    if not ER.exists():
        sys.exit("run this from the repository root (study_tools/eval_run.py not found)")
    text = ER.read_text()
    if MARK in text:
        print("eval_run.py already has ResNet-18 support")
    elif text.count(OLD) != 1:
        sys.exit("eval_run.py: load_model is not the expected version; nothing changed. Send me the file.")
    else:
        shutil.copy(ER, ER.with_suffix(".py.bak_before_rn18"))
        ER.write_text(text.replace(OLD, NEW))
        print("eval_run.py: load_model now reads ResNet-18 or ResNet-50 from the checkpoint "
              "(backup: study_tools/eval_run.py.bak_before_rn18)")

    out = Path("study/rn18/manifest.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in (42, 43, 44):
        for cfg, c in CONFIGS.items():
            for obj in ("simclr", "ncl"):
                rid = f"{obj}-resnet18-octmnist-{cfg}-corrected" + ("" if seed == 42 else f"-s{seed}")
                rows.append(dict(run_id=rid, part="rn18", dataset="octmnist", backbone="resnet18", config=cfg,
                                 objective=obj, seed=seed, prob=0.5, **c))
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} runs)")
    print("\nnext (each run is scored once its epoch-99 checkpoint exists; rerun later to pick up new ones):")
    print("  CUDA_VISIBLE_DEVICES=1 python study_tools/eval_run.py --manifest study/rn18/manifest.csv "
          "--master study/rn18/results_master.csv")
    print("  CUDA_VISIBLE_DEVICES=1 python study_tools/eval_spec.py --master study/rn18/results_master.csv "
          "--out study/rn18/results_spec.csv")


if __name__ == "__main__":
    main()
