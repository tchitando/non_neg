#!/usr/bin/env python3
"""OCT Part 5: the seeds the headline table still needs (ResNet-50, both objectives).

    python study_tools/make_oct_extra.py            # from the repository root
    python study_tools/make_oct_extra.py --dry-run

Seeds 43 and 44 for speckle, blur, pixelation, JPEG and A6 (speckle + contrast, crop 0.2): 20 runs. The pixelation
and A6 run yamls already exist (Part 3 and Part 4); the speckle, blur and JPEG ones are made here from the seed-42
yamls by changing only the seed and the run name. Writes study/manifest_part5.csv and study/launch_part5.sh
(the same claims-based launcher as Parts 1-4). Order: every seed-43 run first.
"""
import argparse, csv, re, sys
from pathlib import Path

CFG = Path("MedMNIST/scripts/pretrain/octmnist")
POLS = ["speckle", "blur", "pixelate", "jpeg", "speckle_contrast_crop020"]


def make_seed_yaml(obj, pid, seed):
    src = CFG / f"{obj}_rn50_p_{pid}.yaml"
    dst = CFG / f"{obj}_rn50_p_{pid}_s{seed}.yaml"
    t = src.read_text()
    t2, n1 = re.subn(r'^name:\s*"([^"]+)"\s*$', lambda m: f'name: "{m.group(1)}-s{seed}"', t, flags=re.M)
    t2, n2 = re.subn(r"^seed:\s*42\s*$", f"seed: {seed}", t2, flags=re.M)
    if (n1, n2) != (1, 1):
        sys.exit(f"{src}: expected one name line and one 'seed: 42' line, found {n1} and {n2}")
    return dst, t2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    rows, files = [], {}
    for seed in (43, 44):
        for pid in POLS:
            for obj in ("ncl", "simclr"):
                cfg = f"{obj}_rn50_p_{pid}_s{seed}"
                dst = CFG / f"{cfg}.yaml"
                dst_new, text = make_seed_yaml(obj, pid, seed)
                if dst.exists():
                    if dst.read_text() != text:
                        print(f"note: {dst} exists and differs from a fresh copy; using the existing file")
                else:
                    files[dst_new] = text
                rows.append(dict(run_id=f"{obj}-resnet50-octmnist-{pid}-s{seed}", part="part5", objective=obj,
                                 seed=seed, policy_id=pid, launch_cfg=cfg, priority={43: 1, 44: 2}[seed]))
    for r in rows:
        print(f"P{r['priority']}  {r['run_id']:58s} {'new yaml' if (CFG / (r['launch_cfg'] + '.yaml')) in files else 'yaml exists'}")
    print(f"\n{len(rows)} runs, {len(files)} new yamls")
    if a.dry_run:
        return
    for p, t in files.items():
        p.write_text(t)
    with open("study/manifest_part5.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    sh = Path("study/launch_part5.sh")
    sh.write_text(Path("study/launch_part3.sh").read_text().replace("part3", "part5").replace("Part 3", "Part 5"))
    sh.chmod(0o755)
    print("wrote the yamls, study/manifest_part5.csv and study/launch_part5.sh")


if __name__ == "__main__":
    main()
