#!/usr/bin/env python3
"""Training-dynamics runs (OCT, ResNet-50): the same runs as the screen, with a checkpoint kept every 9 epochs.

    python study_tools/make_dyn_runs.py                         # stem, contrast, natural-image set; NCL and SimCLR
    python study_tools/make_dyn_runs.py --policies stem contrast cifar motion
    python study_tools/make_dyn_runs.py --dry-run

For each policy and objective it copies MedMNIST/scripts/pretrain/octmnist/{obj}_rn50_p_{policy}.yaml to
{obj}_rn50_p_{policy}_dyn.yaml with three changes only: the run name gets "-dyn", checkpoint.frequency 1 -> 9 and
checkpoint.keep_prev False -> True. solo-learn saves when epoch % frequency == 0, so frequency 9 keeps epochs
0, 9, 18, ..., 99 (12 checkpoints, the last one being the usual epoch-99 checkpoint). Seed 42, as the screen.
Writes study/dyn/manifest.csv and study/dyn/launch.sh. Existing files with different content are never overwritten.
Disk: about 12 checkpoints x 6 runs; delete the intermediate ones after eval_dynamics.py has scored them if space is short.
"""
import argparse, csv, re, sys
from pathlib import Path

CFG = Path("MedMNIST/scripts/pretrain/octmnist")
OUT = Path("study/dyn")
FREQ = 9


def sub_once(pattern, repl, text, what):
    new, n = re.subn(pattern, repl, text, flags=re.M)
    if n != 1:
        sys.exit(f"expected exactly one '{what}' line, found {n}")
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=["stem", "contrast", "cifar"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    files, rows = {}, []
    for pid in a.policies:
        for obj in ("ncl", "simclr"):
            src = CFG / f"{obj}_rn50_p_{pid}.yaml"
            if not src.exists():
                sys.exit(f"missing {src}")
            t = src.read_text()
            name = f"{obj}-resnet50-octmnist-{pid}"
            t = sub_once(rf'^name:\s*"{re.escape(name)}"\s*$', f'name: "{name}-dyn"', t, "name")
            t = sub_once(r"^(\s+)frequency:\s*1\s*$", rf"\g<1>frequency: {FREQ}", t, "checkpoint.frequency: 1")
            t = sub_once(r"^(\s+)keep_prev:\s*False\s*$", r"\g<1>keep_prev: True", t, "checkpoint.keep_prev: False")
            cfg = f"{obj}_rn50_p_{pid}_dyn"
            files[CFG / f"{cfg}.yaml"] = t
            rows.append(dict(run_id=f"{name}-dyn", objective=obj, policy_id=pid, seed=42, launch_cfg=cfg))
    rows.sort(key=lambda r: (a.policies.index(r["policy_id"]), r["objective"]))
    lines = [r["run_id"] + "," + r["launch_cfg"] for r in rows]
    files[OUT / "launch.sh"] = LAUNCH
    for p, t in files.items():
        if p.exists() and p.read_text() != t:
            sys.exit(f"{p} exists with different content; not overwriting")
    print(f"{len(rows)} runs:"); [print("  ", r["run_id"], "<-", r["launch_cfg"] + ".yaml") for r in rows]
    if a.dry_run:
        print("dry run: nothing written"); return
    OUT.mkdir(parents=True, exist_ok=True)
    for p, t in files.items():
        p.write_text(t)
    (OUT / "launch.sh").chmod(0o755)
    with open(OUT / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} configs, {OUT}/manifest.csv, {OUT}/launch.sh")


LAUNCH = r'''#!/usr/bin/env bash
# Training-dynamics launcher (checkpoints kept every 9 epochs). Several copies can run at once; each run is claimed
# with an atomic mkdir in study/claims/, and a run counts as done only when its epoch-99 checkpoint exists.
#   CUDA_VISIBLE_DEVICES=1 bash study/dyn/launch.sh
#   CUDA_VISIBLE_DEVICES=0 bash study/dyn/launch.sh '^ncl-'      # optional regex filter on run ids
# If a run was interrupted, remove its claim to retry it:  rmdir study/claims/<run_id>
cd "$(dirname "$0")/../.." || exit 1
DONE=study/dyn/done.txt; TIMING=study/dyn/timing.csv; CLAIMS=study/claims
mkdir -p "$CLAIMS"; touch "$DONE"; [ -f "$TIMING" ] || echo "run_id,seconds,gpu_hours,gpu" > "$TIMING"
PAT="${1:-.}"
python3 - <<'PY' > /tmp/dyn_queue_$$.txt
import csv
for r in csv.DictReader(open("study/dyn/manifest.csv")):
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
    echo "NOT FINISHED $RUN (no epoch-99 checkpoint); claim kept, remove it to retry" | tee -a study/dyn/failed.txt
  fi
done < /tmp/dyn_queue_$$.txt
rm -f /tmp/dyn_queue_$$.txt
echo "launcher finished"
'''

if __name__ == "__main__":
    main()
