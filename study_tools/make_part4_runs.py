#!/usr/bin/env python3
"""Part 4: the A6-equivalent policy on ResNet-50, three seeds per objective.

    python study_tools/make_part4_runs.py            # writes the files
    python study_tools/make_part4_runs.py --dry-run  # lists the runs only

A6 (FINDINGS.md section 14) is the stem with speckle (p 0.5, std 0.1) and contrast reduction (p 0.5, factor 0.8)
at crop minimum area 0.2. In the current pipeline that is p_speckle_contrast.yaml with crop_min_scale 0.2;
everything else (run settings, operation order, strengths) is unchanged.
New policy id: speckle_contrast_crop020. Seeds 42, 43, 44 for NCL and SimCLR.

Outputs: MedMNIST/scripts/pretrain/octmnist/augmentations/p_speckle_contrast_crop020.yaml,
         MedMNIST/scripts/pretrain/octmnist/{ncl,simclr}_rn50_p_speckle_contrast_crop020[_s43|_s44].yaml,
         study/manifest_part4.csv, study/launch_part4.sh, study/part4_files.tgz
"""
import argparse, csv, re, tarfile
from pathlib import Path

OCT = Path("MedMNIST/scripts/pretrain/octmnist")
OBJS = ["ncl", "simclr"]
BASE_POL, CROP, NEW_PID = "speckle_contrast", 0.2, "speckle_contrast_crop020"
SEEDS = [42, 43, 44]


def sub_once(pat, repl, text, what, flags=re.M):
    new, n = re.subn(pat, repl, text, flags=flags)
    if n != 1:
        raise SystemExit(f"expected exactly one '{what}' line, found {n}")
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    man2 = {r["run_id"]: r for r in csv.DictReader(open("study/manifest_part2.csv"))}
    files, rows = {}, []
    aug_file = f"p_{NEW_PID}.yaml"
    t = (OCT / "augmentations" / f"p_{BASE_POL}.yaml").read_text()
    files[OCT / "augmentations" / aug_file] = sub_once(r"^(\s*crop_min_scale:\s*)\S+\s*$", rf"\g<1>{CROP}", t, "crop_min_scale")
    for seed in SEEDS:                      # launch order: all seed-42 runs first
        for obj in OBJS:
            src = man2[f"{obj}-resnet50-octmnist-{BASE_POL}"]
            t = (OCT / f"{obj}_rn50_p_{BASE_POL}.yaml").read_text()
            suffix = "" if seed == 42 else f"-s{seed}"
            run_id = f"{obj}-resnet50-octmnist-{NEW_PID}{suffix}"
            cfg = f"{obj}_rn50_p_{NEW_PID}" + ("" if seed == 42 else f"_s{seed}")
            t = sub_once(r"^(\s*-\s*augmentations:\s*)\S+\s*$", rf"\g<1>{aug_file}", t, "augmentations")
            t = sub_once(r'^name:\s*".*"\s*$', f'name: "{run_id}"', t, "name")
            t = sub_once(r"^seed:\s*\d+\s*$", f"seed: {seed}", t, "seed")
            files[OCT / f"{cfg}.yaml"] = t
            rows.append(dict(run_id=run_id, part=4, objective=obj, seed=seed, policy_id=NEW_PID, bitmask=src["bitmask"],
                             augs_on=src["augs_on"], n_augs=src["n_augs"], prob=src["prob"], levels=src["levels"],
                             crop_min_scale=CROP, aug_yaml=aug_file, run_yaml=f"{cfg}.yaml", launch_cfg=cfg,
                             priority=1, kind="a6"))
    for r in rows:
        print(f"{r['run_id']:56s} seed {r['seed']}  crop {r['crop_min_scale']}  -> {r['launch_cfg']}")
    print(f"\n{len(rows)} runs, {len(files)} new files")
    if a.dry_run:
        return
    for p, txt in files.items():
        if p.exists() and p.read_text() != txt:
            raise SystemExit(f"{p} exists with different content; not overwriting")
        p.write_text(txt)
    with open("study/manifest_part4.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    Path("study/launch_part4.sh").write_text(LAUNCH)
    with tarfile.open("study/part4_files.tgz", "w:gz") as tf:
        for p in list(files) + [Path("study/manifest_part4.csv"), Path("study/launch_part4.sh")]:
            tf.add(p)
    print("wrote the yamls, study/manifest_part4.csv, study/launch_part4.sh and study/part4_files.tgz")


LAUNCH = r'''#!/usr/bin/env bash
# Part 4 launcher (A6-equivalent policy). Same claim logic as Part 3: several copies can run at once,
# each run is claimed with an atomic mkdir in study/claims/, and a run counts as done only when its
# epoch-99 checkpoint exists.
#   CUDA_VISIBLE_DEVICES=1 bash study/launch_part4.sh
# If a run was interrupted, remove its claim to retry it:  rmdir study/claims/<run_id>
cd "$(dirname "$0")/.." || exit 1
DONE=study/done_part4.txt; TIMING=study/timing_part4.csv; CLAIMS=study/claims
mkdir -p "$CLAIMS"; touch "$DONE"; [ -f "$TIMING" ] || echo "run_id,seconds,gpu_hours,gpu" > "$TIMING"
PAT="${1:-.}"
python3 - <<'PY' > /tmp/part4_queue_$$.txt
import csv
for r in csv.DictReader(open("study/manifest_part4.csv")):
    print(r["run_id"] + "," + r["launch_cfg"])
PY
while IFS=, read -r RUN CFG; do
  echo "$RUN" | grep -Eq "$PAT" || continue
  grep -qx "$RUN" "$DONE" && continue
  mkdir "$CLAIMS/$RUN" 2>/dev/null || continue
  echo "=== $RUN  (GPU ${CUDA_VISIBLE_DEVICES:-?})"; T0=$(date +%s)
  python3 MedMNIST/main_pretrain.py --config-path MedMNIST/scripts/pretrain/octmnist --config-name "$CFG"
  if find trained_models -name "$RUN-offline-*-ep=99.ckpt" 2>/dev/null | grep -q .; then
    S=$(( $(date +%s) - T0 )); echo "$RUN" >> "$DONE"
    echo "$RUN,$S,$(python3 -c "print(round($S/3600,3))"),${CUDA_VISIBLE_DEVICES:-?}" >> "$TIMING"
  else
    echo "NOT FINISHED $RUN (no epoch-99 checkpoint); claim kept, remove it to retry" | tee -a study/failed_part4.txt
  fi
done < /tmp/part4_queue_$$.txt
rm -f /tmp/part4_queue_$$.txt
echo "launcher finished"
'''

if __name__ == "__main__":
    main()
