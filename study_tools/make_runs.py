#!/usr/bin/env python3
"""Generate augmentation yamls, run yamls, a manifest and a launch script for the OCT augmentation study.

Run from the repository root (the folder that contains MedMNIST/ and scripts/ is assumed to be
`MedMNIST/scripts/pretrain/octmnist`; change --octdir if yours differs).

    python make_runs.py --part 1 --cifar-yaml path/to/cifar_policy.yaml
    python make_runs.py --part 2
    python make_runs.py --part 3

What it writes (nothing existing is modified):
    <octdir>/augmentations/p_<policy>.yaml     one augmentation file per policy
    <octdir>/{ncl,simclr}_rn50_p_<policy>[_s43].yaml   one run file per run
    study/manifest_part<N>.csv                 one row per planned run
    study/launch_part<N>.sh                    runs the list, skips finished runs, logs wall time
    study/levels.json                          strengths per augmentation (created once, then yours)

The fixed stem (random resized crop min scale 0.08, horizontal flip 0.5, brightness jitter) and the
whole training recipe come from the A0 files and are never touched.
"""
import argparse, csv, itertools, json, re, sys
from pathlib import Path

POOL = ["speckle", "contrast", "blur", "motion", "pixelate", "jpeg"]   # fixed bit order
SEEDS_STEM = [42, 43, 44]

DEFAULT_LEVELS = {
    "_note": "Provisional. speckle/contrast follow the label test. blur/motion/pixelate/jpeg use the gentler 64 px values "
             "(blur sigma ~ U[0.1, sigma_max]; motion radius, sigma, angle; pixelate factor; jpeg quality) until the mild label test settles them. "
             "Old MedMNIST-C style entries {\"level\": 0..4} still work for motion, pixelate, jpeg.",
    "prob": 0.5,
    "speckle":  {"std": 0.1},
    "contrast": {"factor": 0.8},
    "blur":     {"sigma_max": 1.0},
    "motion":   {"radius": 5, "sigma": 2, "angle": 15},
    "pixelate": {"factor": 0.8},
    "jpeg":     {"quality": 80},
}

AUG_TEMPLATE = """- rrc:
    enabled: True
    crop_min_scale: {crop_min}
    crop_max_scale: 1.0
  color_jitter:
    prob: 0.8
    brightness: 0.8
    contrast: 0.0
    saturation: 0.0
    hue: 0.0
  grayscale:
    prob: 0.0
  gaussian_blur:
    prob: 0.0
  solarization:
    prob: 0.0
  equalization:
    prob: 0.0
  horizontal_flip:
    prob: 0.5
  speckle_noise:
    prob: {speckle_p}
    std: {speckle_std}
  contrast_decrease:
    prob: {contrast_p}
    factor: {contrast_factor}
  blur_custom:
    prob: {blur_p}
    sigma_lo: 0.1
    sigma_hi: {blur_hi}
  motion_blur:
    prob: {motion_p}
{motion_params}  pixelate:
    prob: {pixelate_p}
{pixelate_params}  jpeg_compression:
    prob: {jpeg_p}
{jpeg_params}  crop_size: 64
  num_crops: 2
  mean: [0.1894, 0.1894, 0.1894]
  std: [0.2071, 0.2071, 0.2071]
"""


def policy_id(augs):
    if not augs:
        return "stem"
    if len(augs) == len(POOL):
        return "all6"
    return "_".join(a for a in POOL if a in augs)


def bitmask(augs):
    return "".join("1" if a in augs else "0" for a in POOL)


def _params(lv, keys, indent="    "):
    """Yaml lines for the parameters present in `lv`; MedMNIST-C style {'level': n} is used when no explicit values exist."""
    if all(k in lv for k in keys):
        return "".join(f"{indent}{k}: {lv[k]}\n" for k in keys)
    return f"{indent}level: {lv.get('level', 0)}\n"


def render_aug(augs, levels, crop_min=0.08):
    p = levels["prob"]
    on = lambda a: p if a in augs else 0.0
    return AUG_TEMPLATE.format(
        crop_min=crop_min,
        blur_p=on("blur"), blur_hi=levels["blur"].get("sigma_max", 2.0),
        speckle_p=on("speckle"), speckle_std=levels["speckle"]["std"],
        contrast_p=on("contrast"), contrast_factor=levels["contrast"]["factor"],
        motion_p=on("motion"), motion_params=_params(levels["motion"], ["radius", "sigma", "angle"]),
        pixelate_p=on("pixelate"), pixelate_params=_params(levels["pixelate"], ["factor"]),
        jpeg_p=on("jpeg"), jpeg_params=_params(levels["jpeg"], ["quality"]),
    )


def sub_once(pattern, repl, text, what):
    new, n = re.subn(pattern, repl, text, flags=re.M)
    if n != 1:
        sys.exit(f"template problem: expected exactly one {what} line, found {n}")
    return new


def render_run(template, objective, pid, seed):
    suffix = "" if seed == 42 else f"-s{seed}"
    name = f"{objective}-resnet50-octmnist-{pid}{suffix}"
    t = sub_once(r"^(\s*-\s*)augmentations:\s*\S+\s*$", rf"\1augmentations: p_{pid}.yaml", template, "augmentations")
    t = sub_once(r'^name:\s*".*"\s*$', f'name: "{name}"', t, "name")
    t = sub_once(r"^seed:\s*\d+\s*$", f"seed: {seed}", t, "seed")
    return name, t


def policies_for_part(part):
    n = {1: [0, 1], 2: [2, 5, 6], 3: [3, 4]}[part]
    out = []
    for k in n:
        out += [set(c) for c in itertools.combinations(POOL, k)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--octdir", default="MedMNIST/scripts/pretrain/octmnist")
    ap.add_argument("--study", default="study")
    ap.add_argument("--ncl-template", default=None, help="default: <octdir>/ncl_rn50_A0.yaml")
    ap.add_argument("--simclr-template", default=None, help="default: <octdir>/simclr_rn50_A0.yaml")
    ap.add_argument("--cifar-yaml", default=None, help="the CIFAR-derived policy file (Part 1 only)")
    ap.add_argument("--launch-cmd", default="python3 MedMNIST/main_pretrain.py --config-path {octdir} --config-name {cfg}",
                    help="command for one run; {octdir} and {cfg} are filled in. CHECK THIS against how you start runs.")
    ap.add_argument("--workers-motion", type=int, default=16,
                    help="data-loading workers for policies that include motion blur (about 17 ms per image, so 4 workers "
                         "make an epoch about 7 minutes; the other policies keep the template value)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    octdir, study = Path(a.octdir), Path(a.study)
    augdir = octdir / "augmentations"
    tmpl = {
        "ncl": Path(a.ncl_template or octdir / "ncl_rn50_A0.yaml").read_text(),
        "simclr": Path(a.simclr_template or octdir / "simclr_rn50_A0.yaml").read_text(),
    }
    study.mkdir(parents=True, exist_ok=True)

    lv_path = study / "levels.json"
    if not lv_path.exists():
        lv_path.write_text(json.dumps(DEFAULT_LEVELS, indent=2))
        print(f"wrote provisional {lv_path} (edit it after the label test)")
    levels = json.loads(lv_path.read_text())

    # (policy id, augs, source) for this part
    pols = [(policy_id(s), s, "pool") for s in policies_for_part(a.part)]
    if a.part == 1 and a.cifar_yaml:
        pols.append(("cifar", None, "cifar"))

    files, rows = {}, []
    for pid, augs, src in pols:
        if src == "cifar":
            files[augdir / f"p_{pid}.yaml"] = adapt_cifar(Path(a.cifar_yaml))
        else:
            files[augdir / f"p_{pid}.yaml"] = render_aug(augs, levels)
        seeds = SEEDS_STEM if pid == "stem" else [42]
        for obj in ("ncl", "simclr"):
            for seed in seeds:
                name, text = render_run(tmpl[obj], obj, pid, seed)
                if augs and "motion" in augs and a.workers_motion:
                    text, nsub = re.subn(r"^(\s*num_workers:\s*)\d+", rf"\g<1>{a.workers_motion}", text, flags=re.M)
                    if nsub != 1:
                        sys.exit(f"template problem: expected one num_workers line, found {nsub}")
                cfg = f"{obj}_rn50_p_{pid}" + ("" if seed == 42 else f"_s{seed}")
                files[octdir / f"{cfg}.yaml"] = text
                augs_on = sorted(augs, key=POOL.index) if augs else []
                rows.append(dict(run_id=name, part=a.part, objective=obj, seed=seed, policy_id=pid,
                                 bitmask=bitmask(augs or []) if src == "pool" else "cifar",
                                 augs_on="+".join(augs_on) if src == "pool" else "cifar policy",
                                 n_augs=len(augs_on) if src == "pool" else "",
                                 prob=levels["prob"], levels=json.dumps({k: levels[k] for k in augs_on}) if src == "pool" else "",
                                 crop_min_scale=0.08, aug_yaml=f"p_{pid}.yaml", run_yaml=f"{cfg}.yaml",
                                 launch_cfg=cfg))
    print(f"part {a.part}: {len(pols)} policies, {len(rows)} runs")
    if a.dry_run:
        for r in rows:
            print("  ", r["run_id"], r["bitmask"])
        return
    for p, t in files.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(t)
    man = study / f"manifest_part{a.part}.csv"
    with man.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    sh = study / f"launch_part{a.part}.sh"
    cmd = a.launch_cmd.format(octdir=octdir, cfg="$CFG")
    sh.write_text(f"""#!/usr/bin/env bash
# Runs every planned run of part {a.part} in order. Finished runs are skipped (done_part{a.part}.txt).
# Wall time per run is appended to timing_part{a.part}.csv; the first rows give H (GPU-hours per run).
cd "$(dirname "$0")/.." || exit 1
DONE=study/done_part{a.part}.txt; TIMING=study/timing_part{a.part}.csv
touch "$DONE"; [ -f "$TIMING" ] || echo "run_id,seconds,gpu_hours" > "$TIMING"
PAT="${{1:-.}}"   # optional filter, e.g.  bash study/launch_part{a.part}.sh 'stem|cifar'
while IFS=, read -r RUN CFG; do
  echo "$RUN" | grep -Eq "$PAT" || continue
  grep -qx "$RUN" "$DONE" && {{ echo "skip $RUN"; continue; }}
  echo "=== $RUN"; T0=$(date +%s)
  {cmd} && {{ T1=$(date +%s); S=$((T1-T0)); echo "$RUN" >> "$DONE"
    echo "$RUN,$S,$(python3 -c "print(round($S/3600,3))")" >> "$TIMING"; }} || echo "FAILED $RUN" | tee -a study/failed_part{a.part}.txt
done < <(python3 - <<'PY'
import csv
for r in csv.DictReader(open("study/manifest_part{a.part}.csv")):
    print(r["run_id"] + "," + r["launch_cfg"])
PY
)
""")
    sh.chmod(0o755)
    print(f"wrote {len(files)} yaml files, {man}, {sh}")


def adapt_cifar(path):
    """Take the CIFAR-derived augmentation file and make it valid for 64 px grayscale OCT.

    Only the keys that depend on the dataset are changed: crop_size 64, num_crops 2, OCT mean and std,
    and the three new blocks are added with prob 0. Everything else is kept as the file has it.
    """
    import yaml
    d = yaml.safe_load(path.read_text())
    d = d[0] if isinstance(d, list) else d
    d["crop_size"], d["num_crops"] = 64, 2
    d["mean"], d["std"] = [0.1894] * 3, [0.2071] * 3
    for k in ("speckle_noise", "contrast_decrease", "blur_custom", "motion_blur", "pixelate", "jpeg_compression"):
        d.setdefault(k, {"prob": 0.0})
    print("CIFAR policy adapted: crop_size 64, OCT mean/std; other keys unchanged:",
          {k: v for k, v in d.items() if k not in ("mean", "std")})
    return yaml.safe_dump([d], sort_keys=False)


if __name__ == "__main__":
    main()
