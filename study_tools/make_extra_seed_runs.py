#!/usr/bin/env python3
"""Part 11 (OCT, ResNet-50): extra seeds for single-seed policies that could be the best on validation.

    python study_tools/make_extra_seed_runs.py --dry-run
    python study_tools/make_extra_seed_runs.py                         # priority 1: 3 policies x seeds 43, 44 x 2 methods
    python study_tools/make_extra_seed_runs.py --policies contrast_f060 contrast_pixelate contrast_jpeg   # priority 2

Priority 1 (chosen 4 Nov 2026): single-seed old-stem policies whose validation probe h is within noise of the best
3-seed policy (stem + contrast) for both methods, or above it for SimCLR:
    contrast_blur      stem + contrast 0.8 + blur        (its seeds 43/44 were planned in Part 3 and never trained)
    contrast_f090      stem + contrast 0.9               (also completes the stem contrast sweep)
    contrast_crop020   stem + contrast 0.8, crop 0.2     (also completes the stem + contrast crop sweep)

Each run file is a copy of the policy's seed-42 run file with only the seed and the run name changed (as Part 3 did;
motion policies also get 16 data-loading workers). A run file that already exists is reused when its content is
identical (contrast_blur s43/s44 were written by Part 3) and never overwritten otherwise.

Outputs (never overwriting a file with different content):
    study/manifest_part11.csv     the runs, with the seed-42 run's manifest fields
    study/launch_part11.sh        launcher; several copies can run at once (one per GPU), runs are claimed in
                                  study/claims/ (shared with every other launcher, so no run is trained twice);
                                  a run whose epoch-99 checkpoint already exists is marked done and skipped
    study/test_plan_part11.csv    test plan for these runs, written now, before any of them is trained or scored
"""
import argparse, csv, re
from datetime import date
from pathlib import Path

OCT = Path("MedMNIST/scripts/pretrain/octmnist")
OBJS = ["ncl", "simclr"]
PRIORITY1 = ["contrast_blur", "contrast_f090", "contrast_crop020"]
FIELDS = ["run_id", "part", "objective", "seed", "policy_id", "bitmask", "augs_on", "n_augs", "prob", "levels",
          "crop_min_scale", "aug_yaml", "run_yaml", "launch_cfg", "priority", "kind"]
PLAN_FIELDS = ["run_id", "dataset", "backbone", "objective", "seed", "policy_id", "fixed_on"]


def sub_once(pat, repl, text, what, flags=re.M):
    new, n = re.subn(pat, repl, text, flags=flags)
    if n != 1:
        raise SystemExit(f"expected exactly one '{what}' line, found {n}")
    return new


def write_new(path, text):
    """Write unless the file exists with different content (then stop)."""
    if path.exists():
        if path.read_text() != text:
            raise SystemExit(f"{path} exists with different content; not overwriting")
        return "kept"
    path.write_text(text)
    return "written"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=PRIORITY1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[43, 44])
    ap.add_argument("--part", default="11", help="manifest / launcher / done-file suffix")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src_rows = {}
    for f in sorted(Path("study").glob("manifest_part*.csv")):
        for r in csv.DictReader(open(f)):
            src_rows.setdefault(r["run_id"], (f.name, r))

    files, rows = {}, []
    for pid in a.policies:
        for seed in a.seeds:
            for obj in OBJS:
                base_id = f"{obj}-resnet50-octmnist-{pid}"
                if base_id not in src_rows:
                    raise SystemExit(f"{base_id} (the seed-42 run) is not in any manifest")
                src = src_rows[base_id][1]
                src_yaml = OCT / f"{obj}_rn50_p_{pid}.yaml"
                if not src_yaml.exists():
                    raise SystemExit(f"{src_yaml} not found")
                run_id = f"{base_id}-s{seed}"
                cfg = f"{obj}_rn50_p_{pid}_s{seed}"
                t = src_yaml.read_text()
                t = sub_once(r"^seed:\s*\d+\s*$", f"seed: {seed}", t, "seed")
                t = sub_once(r'^name:\s*".*"\s*$', f'name: "{run_id}"', t, "name")
                if "motion" in pid.split("_"):
                    t = sub_once(r"^(\s*num_workers:\s*)\d+\s*$", r"\g<1>16", t, "num_workers")
                files[OCT / f"{cfg}.yaml"] = t
                row = {k: src.get(k, "") for k in FIELDS}
                row.update(run_id=run_id, part=a.part, seed=seed, run_yaml=f"{cfg}.yaml", launch_cfg=cfg, priority=1, kind="seed")
                rows.append(row)

    print(f"{'run id':58s} {'seed':>4s}  run file")
    for r in rows:
        other = src_rows.get(r["run_id"])
        note = f"   (also listed in {other[0]})" if other else ""
        print(f"{r['run_id']:58s} {r['seed']:>4}  {r['run_yaml']}{note}")
    print(f"\n{len(rows)} runs")
    if a.dry_run:
        print("dry run: nothing written")
        return

    status = {p: write_new(p, t) for p, t in files.items()}
    print(f"run files: {sum(v == 'written' for v in status.values())} written, {sum(v == 'kept' for v in status.values())} "
          f"already there with the same content")

    man = Path(f"study/manifest_part{a.part}.csv")
    plan = Path(f"study/test_plan_part{a.part}.csv")
    sh = Path(f"study/launch_part{a.part}.sh")
    ids = sorted(r["run_id"] for r in rows)
    for p in (man, plan):
        if p.exists() and sorted(r["run_id"] for r in csv.DictReader(open(p))) != ids:
            raise SystemExit(f"{p} exists with other runs; choose another --part")
    if not man.exists():
        with open(man, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    if not plan.exists():
        with open(plan, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=PLAN_FIELDS); w.writeheader()
            for r in rows:
                w.writerow(dict(run_id=r["run_id"], dataset="octmnist", backbone="resnet50", objective=r["objective"],
                                seed=r["seed"], policy_id=r["policy_id"], fixed_on=date.today().isoformat()))
    write_new(sh, LAUNCH.replace("PART", a.part))
    sh.chmod(0o755)
    print(f"wrote {man}, {plan} and {sh}")
    print(f"\nstart (one launcher per GPU):  CUDA_VISIBLE_DEVICES=0 bash {sh}")


LAUNCH = r'''#!/usr/bin/env bash
# Part PART launcher (extra seeds). Several copies can run at once (one per GPU): each run is claimed with an atomic
# mkdir in study/claims/ (shared with every other launcher) before it starts, so no run is ever started twice.
# A run counts as done only when its epoch-99 checkpoint exists; a run whose checkpoint already exists is skipped.
#   export CUDA_DEVICE_ORDER=PCI_BUS_ID; CUDA_VISIBLE_DEVICES=0 bash study/launch_partPART.sh
#   CUDA_VISIBLE_DEVICES=0 bash study/launch_partPART.sh '^ncl-'     # optional regex filter on run ids
# If a run was interrupted, remove its claim to retry it:  rmdir study/claims/<run_id>
cd "$(dirname "$0")/.." || exit 1
DONE=study/done_partPART.txt; TIMING=study/timing_partPART.csv; CLAIMS=study/claims
mkdir -p "$CLAIMS"; touch "$DONE"; [ -f "$TIMING" ] || echo "run_id,seconds,gpu_hours,gpu" > "$TIMING"
PAT="${1:-.}"
python3 - <<'PY' > /tmp/partPART_queue_$$.txt
import csv
for r in csv.DictReader(open("study/manifest_partPART.csv")):
    print(r["run_id"] + "," + r["launch_cfg"])
PY
while IFS=, read -r RUN CFG; do
  echo "$RUN" | grep -Eq "$PAT" || continue
  grep -qx "$RUN" "$DONE" && continue
  if find trained_models -name "$RUN-offline-*-ep=99.ckpt" 2>/dev/null | grep -q .; then
    echo "$RUN" >> "$DONE"; echo "already trained: $RUN"; continue
  fi
  mkdir "$CLAIMS/$RUN" 2>/dev/null || continue
  echo "=== $RUN  (GPU ${CUDA_VISIBLE_DEVICES:-?})"; T0=$(date +%s)
  python3 MedMNIST/main_pretrain.py --config-path MedMNIST/scripts/pretrain/octmnist --config-name "$CFG"
  if find trained_models -name "$RUN-offline-*-ep=99.ckpt" 2>/dev/null | grep -q .; then
    S=$(( $(date +%s) - T0 )); echo "$RUN" >> "$DONE"
    echo "$RUN,$S,$(python3 -c "print(round($S/3600,3))"),${CUDA_VISIBLE_DEVICES:-?}" >> "$TIMING"
  else
    echo "NOT FINISHED $RUN (no epoch-99 checkpoint); claim kept, remove it to retry" | tee -a study/failed_partPART.txt
  fi
done < /tmp/partPART_queue_$$.txt
rm -f /tmp/partPART_queue_$$.txt
echo "launcher finished"
'''

if __name__ == "__main__":
    main()
