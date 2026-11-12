#!/usr/bin/env python3
"""Part 6 (strength curves, OCT, ResNet-50): run files, manifest and launcher, in priority order.

    python study_tools/make_strength_runs.py --dry-run      # list the runs only
    python study_tools/make_strength_runs.py                # write the files

Three strengths per transform, 3 seeds each, NCL and SimCLR. Levels are set by the label test
(study/label_test_aug): mild ~1 point lost, moderate ~2 points (the level every operation already uses), strong
clearly damaging (~5 points, or one class losing more than 6).

    transform        mild            moderate (exists)     strong
    contrast factor  0.9 (exists)    0.8                   0.6 (exists)
    crop min area    0.5 (exists)    0.2 (exists)          0.08 (the stem)
    speckle std      0.05            0.1                   0.2
    blur sigma max   0.6             1.2                   2.0 (SimCLR's default range 0.1-2.0)

Priority 1: seeds 43, 44 for the contrast and crop levels that have one seed (completes two 3-seed curves).
Priority 2: the new speckle and blur levels, seed 42 (completes two 1-seed curves).
Priority 3: seeds 43, 44 for the new blur levels. Speckle stays at one seed per new level: it is in the strength
screen, not in the focused design (contrast/stain, blur, rotation, crop). Seeds 43, 44 at the moderate level (speckle, blur)
come from Part 5 (make_oct_extra.py), which also seeds pixelation, JPEG and A6; run that first or alongside.
Every run file is a copy of an existing Part 1-3 file with only the seed, the run name and, for a new level,
the augmentation file changed; a new augmentation file differs from its source in one value.
Outputs: study/manifest_part6.csv (picked up by eval_run.py with the other parts), study/launch_part6.sh,
the new yamls under MedMNIST/scripts/pretrain/octmnist/. Existing files with different content are never overwritten.
"""
import argparse, csv, json, re
from pathlib import Path

OCT = Path("MedMNIST/scripts/pretrain/octmnist")
OBJS = ["ncl", "simclr"]
LEVEL = {  # transform -> (augmentation-file pattern for the value, levels.json key, policy-id tag)
    "speckle": (r"(speckle_noise:\s*\n\s*prob:\s*\S+\s*\n\s*std:\s*)\S+", "std", lambda v: f"std{int(round(v * 100)):03d}"),
    "blur": (r"(blur_custom:\s*\n\s*prob:\s*\S+\s*\n\s*sigma_lo:\s*\S+\s*\n\s*sigma_hi:\s*)\S+", "sigma_max",
             lambda v: f"sig{int(round(v * 100)):03d}"),
}
# (priority, kind, source policy, new level or None, seeds)
PLAN = [
    (1, "seed", "contrast_f090", None, [43, 44]),
    (1, "seed", "contrast_f060", None, [43, 44]),
    (1, "seed", "stem_crop020", None, [43, 44]),
    (1, "seed", "stem_crop050", None, [43, 44]),
    (2, "level", "speckle", 0.05, [42]),
    (2, "level", "speckle", 0.2, [42]),
    (2, "level", "blur", 0.6, [42]),
    (2, "level", "blur", 2.0, [42]),
    (3, "level", "blur", 0.6, [43, 44]),
    (3, "level", "blur", 2.0, [43, 44]),
]


def sub_once(pat, repl, text, what, flags=re.M):
    new, n = re.subn(pat, repl, text, flags=flags)
    if n != 1:
        raise SystemExit(f"expected exactly one '{what}' match, found {n}")
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    src_rows = {r["run_id"]: r for f in sorted(Path("study").glob("manifest_part[1-5].csv")) for r in csv.DictReader(open(f))}
    files, rows = {}, []
    for prio, kind, pol, val, seeds in PLAN:
        if kind == "level":
            pat, key, tag = LEVEL[pol]
            pid = f"{pol}_{tag(val)}"
            aug_file = f"p_{pid}.yaml"
            t = (OCT / "augmentations" / f"p_{pol}.yaml").read_text()
            files[OCT / "augmentations" / aug_file] = sub_once(pat, rf"\g<1>{val}", t, f"{pol} {key}", flags=0)
        else:
            pid, aug_file = pol, f"p_{pol}.yaml"
        for seed in seeds:
            for obj in OBJS:
                base_id = f"{obj}-resnet50-octmnist-{pol}"
                if base_id not in src_rows:
                    raise SystemExit(f"{base_id} not in the Part 1-5 manifests")
                src = src_rows[base_id]
                run_id = f"{obj}-resnet50-octmnist-{pid}" + ("" if seed == 42 else f"-s{seed}")
                cfg = f"{obj}_rn50_p_{pid}" + ("" if seed == 42 else f"_s{seed}")
                t = (OCT / f"{obj}_rn50_p_{pol}.yaml").read_text()
                t = sub_once(r"^(\s*-\s*augmentations:\s*)\S+\s*$", rf"\g<1>{aug_file}", t, "augmentations")
                t = sub_once(r'^name:\s*".*"\s*$', f'name: "{run_id}"', t, "name")
                t = sub_once(r"^seed:\s*\d+\s*$", f"seed: {seed}", t, "seed")
                files[OCT / f"{cfg}.yaml"] = t
                levels = src["levels"]
                if kind == "level":
                    lv = json.loads(levels); lv[pol][LEVEL[pol][1]] = val; levels = json.dumps(lv)
                rows.append(dict(run_id=run_id, part=6, objective=obj, seed=seed, policy_id=pid, bitmask=src["bitmask"],
                                 augs_on=src["augs_on"], n_augs=src["n_augs"], prob=src["prob"], levels=levels,
                                 crop_min_scale=src["crop_min_scale"], aug_yaml=aug_file, run_yaml=f"{cfg}.yaml",
                                 launch_cfg=cfg, priority=prio, kind=kind))
    ids = [r["run_id"] for r in rows]
    dup = sorted({i for i in ids if ids.count(i) > 1} | {i for i in ids if i in src_rows})
    if dup:
        raise SystemExit(f"run ids already planned or trained: {dup}")
    for r in rows:
        print(f"P{r['priority']}  {r['run_id']:52s} seed {r['seed']}  {r['levels']:32s} crop {r['crop_min_scale']}  -> {r['launch_cfg']}")
    print(f"\n{len(rows)} runs, {len(files)} new files")
    if a.dry_run:
        print("dry run: nothing written"); return
    for p, t in files.items():
        if p.exists() and p.read_text() != t:
            raise SystemExit(f"{p} exists with different content; not overwriting")
    for p, t in files.items():
        p.write_text(t)
    with open("study/manifest_part6.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    Path("study/launch_part6.sh").write_text(LAUNCH)
    Path("study/launch_part6.sh").chmod(0o755)
    print("wrote study/manifest_part6.csv, study/launch_part6.sh and the yamls")


LAUNCH = r'''#!/usr/bin/env bash
# Part 6 launcher (strength curves). Several copies can run at once: each run is claimed with an atomic mkdir in
# study/claims/, and a run counts as done only when its epoch-99 checkpoint exists. Runs start in priority order.
#   CUDA_VISIBLE_DEVICES=1 bash study/launch_part6.sh
#   CUDA_VISIBLE_DEVICES=0 bash study/launch_part6.sh '^ncl-'    # optional regex filter on run ids
# If a run was interrupted, remove its claim to retry it:  rmdir study/claims/<run_id>
cd "$(dirname "$0")/.." || exit 1
DONE=study/done_part6.txt; TIMING=study/timing_part6.csv; CLAIMS=study/claims
mkdir -p "$CLAIMS"; touch "$DONE"; [ -f "$TIMING" ] || echo "run_id,seconds,gpu_hours,gpu" > "$TIMING"
PAT="${1:-.}"
python3 - <<'PY' > /tmp/part6_queue_$$.txt
import csv
for r in csv.DictReader(open("study/manifest_part6.csv")):
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
    echo "NOT FINISHED $RUN (no epoch-99 checkpoint); claim kept, remove it to retry" | tee -a study/failed_part6.txt
  fi
done < /tmp/part6_queue_$$.txt
rm -f /tmp/part6_queue_$$.txt
echo "launcher finished"
'''

if __name__ == "__main__":
    main()
