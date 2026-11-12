#!/usr/bin/env python3
"""PathMNIST study: augmentation yamls, run yamls, manifest and launcher (52 runs).

    python study_tools/make_path_runs.py              # needs study/path/label_test/levels.json
    python study_tools/make_path_runs.py --dry-run

Design (per objective, NCL and SimCLR):
    stem, the six single operations and the natural-image reference   seeds 42, 43, 44   (24 runs)
    all six operations                                                 seed 42            (1 run)
    stem with crop minimum area 0.2                                    seed 42            (1 run)
Launch order: every seed-42 run first (stem first), then seed 43, then seed 44, so a complete one-seed screen
exists after the first third.

The stem and the training recipe are the OCT study's (ResNet-50, LARS lr 0.4, batch 256, 100 epochs, tau 0.2,
projector 2048-2048-256, NCL = rep_relu); only the dataset, the number of classes, the normalisation and the pool
change. Pool operations and levels: study_tools/path_pool.py and the label test.

Writes MedMNIST/scripts/pretrain/pathmnist_study/{augmentations/p_*.yaml, {ncl,simclr}_rn50_p_*.yaml, wandb/},
study/path/manifest.csv and study/path/launch.sh. Existing files with different content are never overwritten.
"""
import argparse, csv, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import path_pool as PP

OCT = Path("MedMNIST/scripts/pretrain/octmnist")
OUT = Path("MedMNIST/scripts/pretrain/pathmnist_study")
STUDY = Path("study/path")
SEEDS3, OBJS = [42, 43, 44], ["ncl", "simclr"]
MEAN, STD = [0.7406, 0.5331, 0.7059], [0.1292, 0.1765, 0.1381]
# Values written into the yaml block of an operation that is OFF in a policy (prob 0, so never applied). Fixed, so that
# a policy file depends only on the levels of its own operations (the stem files never change when levels do).
OFF_VALUE = {"contrast": 0.8, "rotflip": 1, "stain": 0.05, "defocus": 1.5, "pixelate": 0.8, "jpeg": 70}

STEM = """- rrc:
    enabled: True
    crop_min_scale: {crop}
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
    prob: 0.0
    std: 0.1
{pool}  blur_custom:
    prob: 0.0
    sigma_lo: 0.1
    sigma_hi: 1.2
  motion_blur:
    prob: 0.0
    radius: 5
    sigma: 2
    angle: 15
  crop_size: 64
  num_crops: 2
  mean: {mean}
  std: {std}
"""

CIFAR = """- rrc:
    enabled: True
    crop_min_scale: 0.08
    crop_max_scale: 1.0
  color_jitter:
    prob: 0.8
    brightness: 0.8
    contrast: 0.8
    saturation: 0.8
    hue: 0.2
  grayscale:
    prob: 0.2
  gaussian_blur:
    prob: 0.5
  solarization:
    prob: 0.0
  equalization:
    prob: 0.0
  horizontal_flip:
    prob: 0.5
  speckle_noise:
    prob: 0.0
  contrast_decrease:
    prob: 0.0
  rot_flip:
    prob: 0.0
  stain_jitter:
    prob: 0.0
  blur_custom:
    prob: 0.0
  defocus_blur:
    prob: 0.0
  motion_blur:
    prob: 0.0
  pixelate:
    prob: 0.0
  jpeg_compression:
    prob: 0.0
  crop_size: 64
  num_crops: 2
  mean: {mean}
  std: {std}
"""


def render_aug(on, levels, crop=0.08):
    # yaml blocks in the order the dataloader applies them: contrast_decrease, then the extras
    pool = "".join(PP.yaml_block(n, levels[n] if n in on else OFF_VALUE[n], n in on) for n in PP.POOL)
    return STEM.format(crop=crop, pool=pool, mean=MEAN, std=STD)


def sub_once(pat, repl, text, what):
    new, n = re.subn(pat, repl, text, flags=re.M)
    if n != 1:
        sys.exit(f"template problem: expected exactly one '{what}' line, found {n}")
    return new


def render_run(template, run_id, aug_file, seed):
    t = sub_once(r"^(\s*-\s*augmentations:\s*)\S+\s*$", rf"\g<1>{aug_file}", template, "augmentations")
    t = sub_once(r'^name:\s*".*"\s*$', f'name: "{run_id}"', t, "name")
    t = sub_once(r"^seed:\s*\d+\s*$", f"seed: {seed}", t, "seed")
    t = sub_once(r"^(\s*dataset:\s*)octmnist\s*$", r"\g<1>pathmnist", t, "dataset")
    t = sub_once(r"^(\s*num_classes:\s*)4\s*$", r"\g<1>9", t, "num_classes")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="study/path/label_test/levels.json")
    ap.add_argument("--stem-only", action="store_true",
                    help="only the stem runs (they do not depend on the label test); rerun without it afterwards")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.stem_only:
        levels = {}
        print("stem only: no levels needed")
    else:
        levels = json.loads(Path(a.levels).read_text())
        missing = [n for n in PP.POOL if levels.get(n) is None]
        if missing:
            sys.exit(f"no passing level for {missing} in {a.levels}: decide what to do before generating runs")
        print("levels:", {n: levels[n] for n in PP.POOL})

    # (policy id, operations on, crop, seeds)
    pols = [("stem", set(), 0.08, SEEDS3)] + [(n, {n}, 0.08, SEEDS3) for n in PP.POOL] + [("cifar", None, 0.08, SEEDS3)] \
        + [("all6", set(PP.POOL), 0.08, [42]), ("stem_crop020", set(), 0.2, [42])]
    if a.stem_only:
        pols = pols[:1]
    tmpl = {o: (OCT / f"{o}_rn50_p_stem.yaml").read_text() for o in OBJS}
    files = {OUT / "wandb" / "private.yaml": (OCT / "wandb" / "private.yaml").read_text()}
    rows = []
    for pid, on, crop, seeds in pols:
        aug_file = f"p_{pid}.yaml"
        files[OUT / "augmentations" / aug_file] = (CIFAR.format(mean=MEAN, std=STD) if on is None
                                                   else render_aug(on, levels, crop))
        for seed in seeds:
            for obj in OBJS:
                suffix = "" if seed == 42 else f"-s{seed}"
                run_id = f"{obj}-resnet50-pathmnist-{pid}{suffix}"
                cfg = f"{obj}_rn50_p_{pid}" + ("" if seed == 42 else f"_s{seed}")
                files[OUT / f"{cfg}.yaml"] = render_run(tmpl[obj], run_id, aug_file, seed)
                ops = [] if on is None else [n for n in PP.POOL if n in on]
                rows.append(dict(
                    run_id=run_id, part="path", dataset="pathmnist", objective=obj, seed=seed, policy_id=pid,
                    bitmask="cifar" if on is None else "".join("1" if n in on else "0" for n in PP.POOL),
                    augs_on="cifar policy" if on is None else "+".join(ops), n_augs="" if on is None else len(ops),
                    prob=PP.PROB, levels="" if on is None else json.dumps({n: levels[n] for n in ops}),
                    crop_min_scale=crop, aug_yaml=aug_file, run_yaml=f"{cfg}.yaml", launch_cfg=cfg,
                    priority={42: 1, 43: 2, 44: 3}[seed]))
    rows.sort(key=lambda r: (r["priority"], 0 if r["policy_id"] == "stem" else 1))   # stable: keeps policy order
    for r in rows:
        print(f"P{r['priority']}  {r['run_id']:52s} {r['bitmask']:7s} crop {r['crop_min_scale']}")
    print(f"\n{len(rows)} runs, {len(files)} files")
    if a.dry_run:
        return
    for p, t in files.items():
        if p.exists() and p.read_text() != t:
            sys.exit(f"{p} exists with different content; not overwriting (remove it if the levels changed)")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(t)
    STUDY.mkdir(parents=True, exist_ok=True)
    with (STUDY / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    sh = STUDY / "launch.sh"
    sh.write_text(LAUNCH); sh.chmod(0o755)
    print(f"wrote {OUT}/, {STUDY / 'manifest.csv'} and {sh}")


LAUNCH = r'''#!/usr/bin/env bash
# PathMNIST launcher. Several copies can run at once; each run is claimed with an atomic mkdir in study/claims/,
# and a run counts as done only when its epoch-99 checkpoint exists.
#   CUDA_VISIBLE_DEVICES=1 bash study/path/launch.sh            # all runs, in priority order
#   CUDA_VISIBLE_DEVICES=1 bash study/path/launch.sh '^ncl-'    # optional regex filter on run ids
# If a run was interrupted, remove its claim to let a launcher retry it:  rmdir study/claims/<run_id>
cd "$(dirname "$0")/../.." || exit 1
DONE=study/path/done.txt; TIMING=study/path/timing.csv; CLAIMS=study/claims
mkdir -p "$CLAIMS"; touch "$DONE"; [ -f "$TIMING" ] || echo "run_id,seconds,gpu_hours,gpu" > "$TIMING"
PAT="${1:-.}"
python3 - <<'PY' > /tmp/path_queue_$$.txt
import csv
for r in csv.DictReader(open("study/path/manifest.csv")):
    print(r["run_id"] + "," + r["launch_cfg"])
PY
while IFS=, read -r RUN CFG; do
  echo "$RUN" | grep -Eq "$PAT" || continue
  grep -qx "$RUN" "$DONE" && continue
  mkdir "$CLAIMS/$RUN" 2>/dev/null || continue
  echo "=== $RUN  (GPU ${CUDA_VISIBLE_DEVICES:-?})"; T0=$(date +%s)
  python3 MedMNIST/main_pretrain.py --config-path MedMNIST/scripts/pretrain/pathmnist_study --config-name "$CFG"
  if find trained_models -name "$RUN-offline-*-ep=99.ckpt" 2>/dev/null | grep -q .; then
    S=$(( $(date +%s) - T0 )); echo "$RUN" >> "$DONE"
    echo "$RUN,$S,$(python3 -c "print(round($S/3600,3))"),${CUDA_VISIBLE_DEVICES:-?}" >> "$TIMING"
  else
    echo "NOT FINISHED $RUN (no epoch-99 checkpoint); claim kept, remove it to retry" | tee -a study/path/failed.txt
  fi
done < /tmp/path_queue_$$.txt
rm -f /tmp/path_queue_$$.txt
echo "launcher finished"
'''

if __name__ == "__main__":
    main()
