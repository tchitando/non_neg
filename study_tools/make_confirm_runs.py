#!/usr/bin/env python3
"""Part 3 (confirmation + strength): run files, manifest and launcher, in priority order.

    python study_tools/make_confirm_runs.py            # writes the files
    python study_tools/make_confirm_runs.py --dry-run  # lists the runs only

Every run file is a copy of an existing Part 1/2 file with exactly one thing changed:
    seed runs       <obj>_rn50_p_<policy>.yaml  -> seed 43 or 44, name <obj>-resnet50-octmnist-<policy>-s<seed>
    crop runs       p_<policy>.yaml             -> crop_min_scale 0.2 / 0.5, or the crop switched off (rrc.enabled False)
    contrast runs   p_contrast.yaml             -> contrast factor 0.6 / 0.9
Seed runs keep their policy_id, so summarise.py averages them with the seed-42 run automatically.
Strength runs get new policy ids (stem_crop020, contrast_crop050, contrast_f060, stem_nocrop, ...).

Outputs: study/manifest_part3.csv, study/launch_part3.sh, the new yamls under MedMNIST/scripts/pretrain/octmnist/.
"""
import argparse, csv, re
from pathlib import Path

OCT = Path("MedMNIST/scripts/pretrain/octmnist")
OBJS = ["ncl", "simclr"]

# (priority, kind, policy, variant) in launch order; seed runs: variant = seed
PLAN = [
    (1, "seed", "motion", 43), (1, "seed", "motion", 44),
    (1, "seed", "contrast", 43), (1, "seed", "contrast", 44),
    (1, "seed", "cifar", 43), (1, "seed", "cifar", 44),
    (1, "contrast", "contrast", 0.6), (1, "contrast", "contrast", 0.9),
    (1, "crop", "stem", 0.2), (1, "crop", "contrast", 0.2),
    (1, "crop", "stem", 0.5), (1, "crop", "contrast", 0.5),
    (2, "seed", "speckle_contrast_blur_pixelate_jpeg", 43), (2, "seed", "speckle_contrast_blur_pixelate_jpeg", 44),
    (2, "seed", "pixelate", 43), (2, "seed", "pixelate", 44),
    (2, "seed", "contrast_blur", 43), (2, "seed", "contrast_blur", 44),
    (3, "crop", "stem", None),          # crop switched off
]


def sub_once(pat, repl, text, what, flags=re.M):
    new, n = re.subn(pat, repl, text, flags=flags)
    if n != 1:
        raise SystemExit(f"expected exactly one '{what}' line, found {n}")
    return new


def tag_crop(v):
    return "nocrop" if v is None else f"crop{int(round(v * 100)):03d}"


def tag_contrast(v):
    return f"f{int(round(v * 100)):03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    man2 = {r["run_id"]: r for f in ("study/manifest_part1.csv", "study/manifest_part2.csv") for r in csv.DictReader(open(f))}
    files, rows = {}, []
    for prio, kind, pol, var in PLAN:
        if kind == "seed":
            new_pid = pol
            aug_file = f"p_{pol}.yaml"
        elif kind == "crop":
            new_pid = f"{pol}_{tag_crop(var)}"
            aug_file = f"p_{new_pid}.yaml"
            t = (OCT / "augmentations" / f"p_{pol}.yaml").read_text()
            if var is None:
                t = sub_once(r"^(\s*enabled:\s*)\S+\s*$", r"\1False", t, "rrc.enabled", flags=re.M | re.I)
            else:
                t = sub_once(r"^(\s*crop_min_scale:\s*)\S+\s*$", rf"\g<1>{var}", t, "crop_min_scale")
            files[OCT / "augmentations" / aug_file] = t
        else:  # contrast factor
            new_pid = f"{pol}_{tag_contrast(var)}"
            aug_file = f"p_{new_pid}.yaml"
            t = (OCT / "augmentations" / f"p_{pol}.yaml").read_text()
            t = sub_once(r"(contrast_decrease:\s*\n\s*prob:\s*\S+\s*\n\s*factor:\s*)\S+", rf"\g<1>{var}", t, "contrast factor", flags=0)
            files[OCT / "augmentations" / aug_file] = t
        for obj in OBJS:
            src_run = OCT / f"{obj}_rn50_p_{pol}.yaml"
            t = src_run.read_text()
            base_id = f"{obj}-resnet50-octmnist-{pol}"
            if base_id not in man2:
                raise SystemExit(f"{base_id} not in the Part 1/2 manifests")
            src = man2[base_id]
            if kind == "seed":
                run_id = f"{base_id}-s{var}"
                cfg = f"{obj}_rn50_p_{pol}_s{var}"
                t = sub_once(r"^seed:\s*\d+\s*$", f"seed: {var}", t, "seed")
                seed = var
            else:
                run_id = f"{obj}-resnet50-octmnist-{new_pid}"
                cfg = f"{obj}_rn50_p_{new_pid}"
                t = sub_once(r"^(\s*-\s*augmentations:\s*)\S+\s*$", rf"\g<1>{aug_file}", t, "augmentations")
                seed = 42
            t = sub_once(r'^name:\s*".*"\s*$', f'name: "{run_id}"', t, "name")
            if "motion" in pol.split("_"):
                # motion blur is CPU-bound: 16 data-loading workers (speed only; Part 1 files predate this fix)
                t = sub_once(r"^(\s*num_workers:\s*)\d+\s*$", r"\g<1>16", t, "num_workers")
            files[OCT / f"{cfg}.yaml"] = t
            crop = src["crop_min_scale"]
            if kind == "crop":
                crop = "1.0 (off)" if var is None else var
            rows.append(dict(run_id=run_id, part=3, objective=obj, seed=seed, policy_id=new_pid, bitmask=src["bitmask"],
                             augs_on=src["augs_on"], n_augs=src["n_augs"], prob=src["prob"],
                             levels=src["levels"] if kind != "contrast" else src["levels"].replace('"factor": 0.8', f'"factor": {var}'),
                             crop_min_scale=crop, aug_yaml=aug_file, run_yaml=f"{cfg}.yaml", launch_cfg=cfg,
                             priority=prio, kind=kind))
    for r in rows:
        print(f"P{r['priority']}  {r['run_id']:58s} {r['kind']:9s} seed {r['seed']}  crop {r['crop_min_scale']}  -> {r['launch_cfg']}")
    print(f"\n{len(rows)} runs, {len(files)} new files")
    if a.dry_run:
        return
    for p, t in files.items():
        if p.exists() and p.read_text() != t:
            raise SystemExit(f"{p} exists with different content; not overwriting")
        p.write_text(t)
    with open("study/manifest_part3.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    Path("study/launch_part3.sh").write_text(LAUNCH)
    print("wrote study/manifest_part3.csv, study/launch_part3.sh and the yamls")


LAUNCH = r'''#!/usr/bin/env bash
# Part 3 launcher. Several copies can run at once (e.g. three on GPU 1, one on GPU 0): each run is claimed
# with an atomic mkdir in study/claims/ before it starts, so no run is ever started twice.
# A run counts as done only when its epoch-99 checkpoint exists (a Ctrl-C can no longer mark a run done).
#   CUDA_VISIBLE_DEVICES=1 bash study/launch_part3.sh            # all runs, in priority order
#   CUDA_VISIBLE_DEVICES=0 bash study/launch_part3.sh '^ncl-'    # optional regex filter
# If a run was interrupted, remove its claim to let a launcher pick it up again:  rmdir study/claims/<run_id>
cd "$(dirname "$0")/.." || exit 1
DONE=study/done_part3.txt; TIMING=study/timing_part3.csv; CLAIMS=study/claims
mkdir -p "$CLAIMS"; touch "$DONE"; [ -f "$TIMING" ] || echo "run_id,seconds,gpu_hours,gpu" > "$TIMING"
PAT="${1:-.}"
python3 - <<'PY' > /tmp/part3_queue_$$.txt
import csv
for r in csv.DictReader(open("study/manifest_part3.csv")):
    print(r["run_id"] + "," + r["launch_cfg"])
PY
while IFS=, read -r RUN CFG; do
  echo "$RUN" | grep -Eq "$PAT" || continue
  grep -qx "$RUN" "$DONE" && continue
  mkdir "$CLAIMS/$RUN" 2>/dev/null || continue          # another launcher has it
  echo "=== $RUN  (GPU ${CUDA_VISIBLE_DEVICES:-?})"; T0=$(date +%s)
  python3 MedMNIST/main_pretrain.py --config-path MedMNIST/scripts/pretrain/octmnist --config-name "$CFG"
  if find trained_models -name "$RUN-offline-*-ep=99.ckpt" 2>/dev/null | grep -q .; then
    S=$(( $(date +%s) - T0 )); echo "$RUN" >> "$DONE"
    echo "$RUN,$S,$(python3 -c "print(round($S/3600,3))"),${CUDA_VISIBLE_DEVICES:-?}" >> "$TIMING"
  else
    echo "NOT FINISHED $RUN (no epoch-99 checkpoint); claim kept, remove it to retry" | tee -a study/failed_part3.txt
  fi
done < /tmp/part3_queue_$$.txt
rm -f /tmp/part3_queue_$$.txt
echo "launcher finished"
'''

if __name__ == "__main__":
    main()
