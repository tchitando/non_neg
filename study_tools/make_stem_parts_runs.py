#!/usr/bin/env python3
"""Part 9 (OCT, ResNet-50): which half of the stem matters, brightness jitter or flip?

    python study_tools/make_stem_parts_runs.py --dry-run
    python study_tools/make_stem_parts_runs.py

The old stem is crop and resize (minimum area 0.08) + brightness jitter (0.8, p 0.8) + horizontal flip (p 0.5). On
validation, crop only (Part 7) loses 5.1 backbone-probe points for NCL and 2.6 for SimCLR, and NCL's SC above shuffle
falls from 28.5 to 21.5. Two policies split the stem, 3 seeds, NCL and SimCLR (12 runs):
    croponly_bright   crop and resize + brightness jitter (as in the stem), no flip
    croponly_flip     crop and resize + horizontal flip (as in the stem), no brightness jitter
Prediction (the intensity shortcut): brightness jitter restores most of the specialisation and flip does not.
Each augmentation file is p_stem.yaml with one probability set to 0; each run file is a copy of the stem run file
with only the augmentation file, name and seed changed. Seed 42 of both policies runs first.
Outputs: study/manifest_part9.csv, study/launch_part9.sh and the yamls. Existing files with different content are
never overwritten.
"""
import argparse, csv, re
from pathlib import Path

OCT = Path("MedMNIST/scripts/pretrain/octmnist")
OBJS = ["ncl", "simclr"]
PLAN = [("croponly_bright", "horizontal_flip"), ("croponly_flip", "color_jitter")]   # (new policy, block switched off)
SEEDS = [42, 43, 44]


def sub_once(pat, repl, text, what, flags=re.M):
    new, n = re.subn(pat, repl, text, flags=flags)
    if n != 1:
        raise SystemExit(f"expected exactly one '{what}' match, found {n}")
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    src_rows = {r["run_id"]: r for f in sorted(Path("study").glob("manifest_part*.csv")) if f.name != "manifest_part9.csv"
                for r in csv.DictReader(open(f))}
    files, rows = {}, []
    for seed in SEEDS:
        for pid, off in PLAN:
            pol = "stem"
            aug_file = f"p_{pid}.yaml"
            t = (OCT / "augmentations" / f"p_{pol}.yaml").read_text()
            t = sub_once(rf"({off}:\s*\n\s*prob:\s*)\S+", r"\g<1>0.0", t, f"{off} prob", flags=0)
            files[OCT / "augmentations" / aug_file] = t
            for obj in OBJS:
                base_id = f"{obj}-resnet50-octmnist-{pol}"
                if base_id not in src_rows:
                    raise SystemExit(f"{base_id} not in the earlier manifests")
                src = src_rows[base_id]
                run_id = f"{obj}-resnet50-octmnist-{pid}" + ("" if seed == 42 else f"-s{seed}")
                cfg = f"{obj}_rn50_p_{pid}" + ("" if seed == 42 else f"_s{seed}")
                r = (OCT / f"{obj}_rn50_p_{pol}.yaml").read_text()
                r = sub_once(r"^(\s*-\s*augmentations:\s*)\S+\s*$", rf"\g<1>{aug_file}", r, "augmentations")
                r = sub_once(r'^name:\s*".*"\s*$', f'name: "{run_id}"', r, "name")
                r = sub_once(r"^seed:\s*\d+\s*$", f"seed: {seed}", r, "seed")
                files[OCT / f"{cfg}.yaml"] = r
                rows.append(dict(run_id=run_id, part=9, objective=obj, seed=seed, policy_id=pid, bitmask=src["bitmask"],
                                 augs_on=src["augs_on"], n_augs=src["n_augs"], prob=src["prob"], levels=src["levels"],
                                 crop_min_scale=src["crop_min_scale"], aug_yaml=aug_file, run_yaml=f"{cfg}.yaml",
                                 launch_cfg=cfg, priority=1, kind="stem_parts"))
    dup = sorted(r["run_id"] for r in rows if r["run_id"] in src_rows)
    if dup:
        raise SystemExit(f"run ids already planned: {dup}")
    for r in rows:
        print(f"{r['run_id']:50s} seed {r['seed']}  {r['aug_yaml']:26s} -> {r['launch_cfg']}")
    print(f"\n{len(rows)} runs, {len(files)} new files")
    if a.dry_run:
        print("dry run: nothing written"); return
    for p, t in files.items():
        if p.exists() and p.read_text() != t:
            raise SystemExit(f"{p} exists with different content; not overwriting")
    for p, t in files.items():
        p.write_text(t)
    with open("study/manifest_part9.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    sh = Path("study/launch_part9.sh")
    sh.write_text(LAUNCH)
    sh.chmod(0o755)
    print("wrote study/manifest_part9.csv, study/launch_part9.sh and the yamls")


LAUNCH = r'''#!/usr/bin/env bash
# Part 9 launcher (stem split: brightness, flip). Several copies can run at once: each run is claimed with an atomic mkdir in
# study/claims/, and a run counts as done only when its epoch-99 checkpoint exists. Runs start in priority order.
#   CUDA_VISIBLE_DEVICES=1 bash study/launch_part9.sh
#   CUDA_VISIBLE_DEVICES=0 bash study/launch_part9.sh '^ncl-'    # optional regex filter on run ids
# If a run was interrupted, remove its claim to retry it:  rmdir study/claims/<run_id>
cd "$(dirname "$0")/.." || exit 1
DONE=study/done_part9.txt; TIMING=study/timing_part9.csv; CLAIMS=study/claims
mkdir -p "$CLAIMS"; touch "$DONE"; [ -f "$TIMING" ] || echo "run_id,seconds,gpu_hours,gpu" > "$TIMING"
PAT="${1:-.}"
python3 - <<'PY' > /tmp/part9_queue_$$.txt
import csv
for r in csv.DictReader(open("study/manifest_part9.csv")):
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
    echo "NOT FINISHED $RUN (no epoch-99 checkpoint); claim kept, remove it to retry" | tee -a study/failed_part9.txt
  fi
done < /tmp/part9_queue_$$.txt
rm -f /tmp/part9_queue_$$.txt
echo "launcher finished"
'''

if __name__ == "__main__":
    main()
