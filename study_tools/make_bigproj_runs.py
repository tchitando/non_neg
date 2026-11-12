#!/usr/bin/env python3
"""Part 10 (OCT, ResNet-50): a bigger projector output, everything else unchanged.

    python study_tools/make_bigproj_runs.py --dry-run                          # crop only, output 2048, seeds 42-44
    python study_tools/make_bigproj_runs.py
    python study_tools/make_bigproj_runs.py --only croponly croponly_bright    # also crop + brightness

Question: is NCL's lower projector-probe accuracy (crop only: NCL 68.5 against SimCLR 79.1 on validation) a capacity
limit of the 256-unit projector output? The NCL paper (ImageNet-100) used a 2048-unit output. Here only the output
size changes: projector 2048 -> 2048 -> OUT (default 2048; --hidden changes the hidden layer too, e.g. 16384 to match
the paper). Same augmentation file, epochs, optimiser and seeds as the existing runs of the policy, so each new run
has a 256-unit twin to compare with.

Run ids: <objective>-resnet50-octmnist-<policy>_proj<OUT>[_h<HIDDEN>][-s<seed>]. Run files are copies of the stem run
file with only the augmentation file, name, seed and projector sizes changed. Writes study/manifest_part10.csv (or
manifest_part10b.csv, ... if that name holds other runs) and the matching study/launch_part10*.sh. Existing files
with different content are never overwritten. The scorers read the projector size from the checkpoint.
"""
import argparse, csv, re
from pathlib import Path

OCT = Path("MedMNIST/scripts/pretrain/octmnist")
OBJS = ["ncl", "simclr"]
AUG = {"croponly": "p_croponly.yaml", "croponly_bright": "p_croponly_bright.yaml"}


def sub_once(pat, repl, text, what, flags=re.M):
    new, n = re.subn(pat, repl, text, flags=flags)
    if n != 1:
        raise SystemExit(f"expected exactly one '{what}' match, found {n}")
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="+", default=["croponly"], choices=sorted(AUG))
    ap.add_argument("--out", type=int, default=2048, help="projector output units (now 256)")
    ap.add_argument("--hidden", type=int, default=2048, help="projector hidden units (now 2048)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    a = ap.parse_args()
    tag = f"_proj{a.out}" + ("" if a.hidden == 2048 else f"_h{a.hidden}")
    mans = {f: [r["run_id"] for r in csv.DictReader(open(f))] for f in sorted(Path("study").glob("manifest_part*.csv"))}
    src_rows = {r["run_id"]: r for f in mans for r in csv.DictReader(open(f))}
    files, rows = {}, []
    for seed in a.seeds:
        for base in a.only:
            if not (OCT / "augmentations" / AUG[base]).exists():
                raise SystemExit(f"{AUG[base]} not found: train {base} first")
            pid = base + tag
            for obj in OBJS:
                src = src_rows.get(f"{obj}-resnet50-octmnist-{base}")
                if src is None:
                    raise SystemExit(f"{obj}-resnet50-octmnist-{base} not in the manifests")
                run_id = f"{obj}-resnet50-octmnist-{pid}" + ("" if seed == 42 else f"-s{seed}")
                cfg = f"{obj}_rn50_p_{pid}" + ("" if seed == 42 else f"_s{seed}")
                r = (OCT / f"{obj}_rn50_p_stem.yaml").read_text()
                r = sub_once(r"^(\s*-\s*augmentations:\s*)\S+\s*$", rf"\g<1>{AUG[base]}", r, "augmentations")
                r = sub_once(r'^name:\s*".*"\s*$', f'name: "{run_id}"', r, "name")
                r = sub_once(r"^seed:\s*\d+\s*$", f"seed: {seed}", r, "seed")
                r = sub_once(r"^(\s*proj_output_dim:\s*)\d+\s*$", rf"\g<1>{a.out}", r, "proj_output_dim")
                r = sub_once(r"^(\s*proj_hidden_dim:\s*)\d+\s*$", rf"\g<1>{a.hidden}", r, "proj_hidden_dim")
                files[OCT / f"{cfg}.yaml"] = r
                rows.append(dict(run_id=run_id, part=10, objective=obj, seed=seed, policy_id=pid,
                                 bitmask=src.get("bitmask", ""), augs_on=src.get("augs_on", ""), n_augs=src.get("n_augs", ""),
                                 prob=src.get("prob", ""), levels=src.get("levels", ""), crop_min_scale=src.get("crop_min_scale", ""),
                                 aug_yaml=AUG[base], run_yaml=f"{cfg}.yaml", launch_cfg=cfg, priority=1, kind="projector_size",
                                 proj_hidden_dim=a.hidden, proj_output_dim=a.out))
    new_ids = [r["run_id"] for r in rows]
    same = [f for f, ids in mans.items() if sorted(ids) == sorted(new_ids)]
    if same:
        man = same[0]
    else:
        dup = sorted(i for i in new_ids if i in src_rows)
        if dup:
            raise SystemExit(f"run ids already planned in another manifest: {dup}")
        man = next(Path(f"study/manifest_part10{s}.csv") for s in [""] + list("bcdefghij")
                   if not Path(f"study/manifest_part10{s}.csv").exists())
    for r in rows:
        print(f"{r['run_id']:58s} seed {r['seed']}  {r['aug_yaml']:24s} projector 2048 -> {a.hidden} -> {a.out}")
    print(f"\n{len(rows)} runs, {len(files)} run files; manifest {man}")
    if a.dry_run:
        print("dry run: nothing written"); return
    for p, t in files.items():
        if p.exists() and p.read_text() != t:
            raise SystemExit(f"{p} exists with different content; not overwriting")
    for p, t in files.items():
        p.write_text(t)
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    sh = Path(str(man).replace("manifest_part", "launch_part").replace(".csv", ".sh"))
    sh.write_text(LAUNCH.replace("study/manifest_part10.csv", str(man)))
    sh.chmod(0o755)
    print(f"wrote {man}, {sh} and the run files")


LAUNCH = r'''#!/usr/bin/env bash
# Part 10 launcher (projector size). Several copies can run at once: each run is claimed with an atomic mkdir in
# study/claims/, and a run counts as done only when its epoch-99 checkpoint exists.
#   CUDA_VISIBLE_DEVICES=0 bash study/launch_part10.sh
# If a run was interrupted, remove its claim to retry it:  rmdir study/claims/<run_id>
cd "$(dirname "$0")/.." || exit 1
DONE=study/done_part10.txt; TIMING=study/timing_part10.csv; CLAIMS=study/claims
mkdir -p "$CLAIMS"; touch "$DONE"; [ -f "$TIMING" ] || echo "run_id,seconds,gpu_hours,gpu" > "$TIMING"
PAT="${1:-.}"
python3 - <<'PY' > /tmp/part10_queue_$$.txt
import csv
for r in csv.DictReader(open("study/manifest_part10.csv")):
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
    echo "NOT FINISHED $RUN (no epoch-99 checkpoint); claim kept, remove it to retry" | tee -a study/failed_part10.txt
  fi
done < /tmp/part10_queue_$$.txt
rm -f /tmp/part10_queue_$$.txt
echo "launcher finished"
'''

if __name__ == "__main__":
    main()
