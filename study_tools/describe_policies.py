#!/usr/bin/env python3
"""One line per policy saying exactly which augmentations were applied, at what level and with what probability,
read from the augmentation files the runs were trained with.

    python study_tools/describe_policies.py            # from the repository root on the server
    -> study/derived/policy_augs.csv  (dataset, backbone, policy_id, aug_yaml, augmentations, seeds_checked)

Sources: OCT ResNet-50: study/manifest_part*.csv (column aug_yaml) and MedMNIST/scripts/pretrain/octmnist/augmentations;
OCT ResNet-18: study/rn18/manifest.csv (config A0 / A6 -> A0.yaml / A6.yaml in the same folder); PathMNIST:
study/path/manifest.csv and MedMNIST/scripts/pretrain/pathmnist_study/augmentations. Every run of a policy is checked:
if two seeds of one policy were trained with different augmentation files, the line says so.
Example output: "crop 0.08 · brightness 0.8 (p 0.8) · flip (p 0.5) · speckle std 0.1 (p 0.5)".
"""
import csv, glob
from pathlib import Path

import yaml

OCT = Path("MedMNIST/scripts/pretrain/octmnist/augmentations")
PATH = Path("MedMNIST/scripts/pretrain/pathmnist_study/augmentations")
SKIP = {"rrc", "crop_size", "num_crops", "mean", "std"}


def fmt(v):
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def describe(cfg):
    out = []
    rrc = cfg.get("rrc", {})
    out.append(f"crop {fmt(rrc.get('crop_min_scale'))}" if rrc.get("enabled", True) else "no crop (resize only)")
    cj = cfg.get("color_jitter", {})
    if cj.get("prob", 0) > 0:
        b, c, s, h = (cj.get(k, 0) for k in ("brightness", "contrast", "saturation", "hue"))
        if b and not (c or s or h):
            out.append(f"brightness {fmt(b)} (p {fmt(cj['prob'])})")
        else:
            out.append(f"colour jitter {fmt(b)}/{fmt(c)}/{fmt(s)}/{fmt(h)} (p {fmt(cj['prob'])})")
    simple = {"grayscale": "grayscale", "gaussian_blur": "Gaussian blur (SimCLR)", "solarization": "solarization",
              "equalization": "equalization", "horizontal_flip": "flip", "rot_flip": "rotation 90° + vertical flip"}
    for k, name in simple.items():
        p = cfg.get(k, {}).get("prob", 0) if isinstance(cfg.get(k), dict) else 0
        if p > 0:
            out.append(f"{name} (p {fmt(p)})")
    named = {
        "speckle_noise": lambda c: f"speckle std {fmt(c.get('std'))}",
        "contrast_decrease": lambda c: f"contrast {fmt(c.get('factor'))}",
        "stain_jitter": lambda c: f"stain σ {fmt(c.get('sigma'))}",
        "defocus_blur": lambda c: f"defocus radius {fmt(c.get('radius'))}",
        "blur_custom": lambda c: f"blur σ {fmt(c.get('sigma_lo'))}–{fmt(c.get('sigma_hi'))}",
        "motion_blur": lambda c: (f"motion radius {fmt(c['radius'])}, σ {fmt(c.get('sigma'))}, ±{fmt(c.get('angle', 15))}°"
                                  if c.get("radius") is not None else f"motion level {fmt(c.get('level'))}"),
        "pixelate": lambda c: (f"pixelation {fmt(c['factor'])}" if c.get("factor") is not None
                               else f"pixelation level {fmt(c.get('level'))}"),
        "jpeg_compression": lambda c: (f"JPEG quality {fmt(c['quality'])}" if c.get("quality") is not None
                                       else f"JPEG level {fmt(c.get('level'))}"),
    }
    for k, f in named.items():
        c = cfg.get(k)
        if isinstance(c, dict) and c.get("prob", 0) > 0:
            out.append(f"{f(c)} (p {fmt(c['prob'])})")
    known = set(simple) | set(named) | SKIP | {"color_jitter"}
    for k, c in cfg.items():          # anything this script does not know about is still reported
        if k not in known and isinstance(c, dict) and c.get("prob", 0) > 0:
            out.append(f"{k} {', '.join(f'{a} {fmt(b)}' for a, b in c.items() if a != 'prob')} (p {fmt(c['prob'])})")
    return " · ".join(out)


def load(path):
    d = yaml.safe_load(Path(path).read_text())
    return d[0] if isinstance(d, list) else d


def main():
    groups = {}   # (dataset, backbone, policy) -> {aug_yaml: [run ids]}
    for f in sorted(glob.glob("study/manifest_part*.csv")):
        for r in csv.DictReader(open(f)):
            if not r.get("aug_yaml"):      # Part 5 lists seeds only; its policies' files come from the other manifests
                continue
            groups.setdefault(("OCT", "ResNet-50", r["policy_id"]), {}).setdefault(OCT / r["aug_yaml"], []).append(r["run_id"])
    if Path("study/rn18/manifest.csv").exists():
        for r in csv.DictReader(open("study/rn18/manifest.csv")):
            pid = {"A0": "stem", "A6": "speckle_contrast_crop020"}.get(r["config"], r["config"])
            groups.setdefault(("OCT", "ResNet-18", pid), {}).setdefault(OCT / f"{r['config']}.yaml", []).append(r["run_id"])
    if Path("study/path/manifest.csv").exists():
        for r in csv.DictReader(open("study/path/manifest.csv")):
            groups.setdefault(("PathMNIST", "ResNet-50", r["policy_id"]), {}).setdefault(PATH / r["aug_yaml"], []).append(r["run_id"])
    rows, problems = [], 0
    for (ds, bb, pid), files in sorted(groups.items()):
        descs = {}
        for path, rids in files.items():
            try:
                descs.setdefault(describe(load(path)), []).append(path.name)
            except FileNotFoundError:
                descs.setdefault(f"(missing file {path.name})", []).append(path.name)
        text = " | ".join(descs) if len(descs) > 1 else next(iter(descs))
        if len(descs) > 1:
            problems += 1
            text = "SEEDS DIFFER: " + text
        rows.append(dict(dataset=ds, backbone=bb, policy_id=pid, aug_yaml=", ".join(sorted({p.name for p in files})),
                         augmentations=text, seeds_checked=sum(len(v) for v in files.values())))
    out = Path("study/derived/policy_augs.csv"); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{r['dataset']:9s} {r['backbone']:9s} {r['policy_id']:40s} {r['augmentations']}")
    print(f"\n{len(rows)} policies, {sum(r['seeds_checked'] for r in rows)} runs checked; "
          f"{problems} with seeds trained on different files. Wrote {out}")


if __name__ == "__main__":
    main()
