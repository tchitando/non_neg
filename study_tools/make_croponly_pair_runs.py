#!/usr/bin/env python3
"""Part 12 (OCT, ResNet-50): pairs of augmentations on the crop-only base, for the combination heatmap.

    python study_tools/make_croponly_pair_runs.py --dry-run          # the pairs without blur: 2 pairs x 2 methods x 3 seeds = 12 runs
    python study_tools/make_croponly_pair_runs.py
    python study_tools/make_croponly_pair_runs.py --blur 0.6 --pairs bright+blur contrast+blur blur+flip --dry-run
                                                                      # once blur's level is chosen (18 runs)

Base: crop only (random resized crop, minimum area 0.08, back to 64 px, nothing else), as Parts 7 and 8. Each policy
turns on TWO augmentations, each at the strength chosen from its single-augmentation sweep on the same base (the level
that passes the label-preservation screen with the highest validation backbone balanced accuracy, averaged over NCL and
SimCLR and over seeds 42-44; the same level for both methods):
    brightness   colour jitter, brightness s 0.8, p 0.8        (the stem's setting)
    contrast     contrast decrease, factor 0.3, p 0.5
    flip         horizontal flip, p 0.5                         (the stem's setting)
    blur         Gaussian blur, sigma ~ U[0.1, s], p 0.5        (s given with --blur)
brightness + flip is the stem (Part 1), already trained at seeds 42-44, so it is not made again.

Augmentation files are built from p_stem.yaml: every block is switched off except the two augmentations of the pair,
and the script checks that nothing else is on. Run files are copies of the stem run files with only the augmentation
file, name and seed changed. Run ids follow the existing pattern (seed 42 without suffix, -s43 / -s44 otherwise).
Outputs: study/manifest_part12.csv (then manifest_part12b.csv, ... ; never overwriting), a matching launcher
study/launch_part12*.sh and the yamls. Existing files with different content are never overwritten, and run ids that
are already planned in another manifest are refused.
"""
import argparse, csv, json, re
from pathlib import Path

OCT = Path("MedMNIST/scripts/pretrain/octmnist")
OBJS = ["ncl", "simclr"]
ORDER = ["bright", "contrast", "blur", "flip"]          # order of the parts in a policy id
POOL = ["speckle", "contrast", "blur", "motion", "pixelate", "jpeg"]   # bitmask order used in every manifest
CONTRAST = 0.3
BRIGHT = 0.8
PAIRS = ["bright+contrast", "bright+blur", "contrast+blur", "contrast+flip", "blur+flip"]   # bright+flip = the stem
DEFAULT = ["bright+contrast", "contrast+flip"]


def sub_once(pat, repl, text, what):
    new, n = re.subn(pat, repl, text)
    if n != 1:
        raise SystemExit(f"expected exactly one '{what}' match, found {n}")
    return new


def value(text, block, key):
    m = re.search(rf"{block}:\s*\n(?:\s*\w+:.*\n)*?\s*{key}:\s*(\S+)", text)
    if not m:
        raise SystemExit(f"{block}.{key} not found")
    return float(m.group(1))


def part_id(t, blur):
    if t == "blur":
        return f"blur_sig{round(blur * 100):03d}"
    return {"bright": "bright", "contrast": "contrast_f030", "flip": "flip"}[t]


def aug_text(src, on, blur):
    t = src
    t = sub_once(r"(color_jitter:\s*\n\s*prob:\s*)\S+", r"\g<1>" + ("0.8" if "bright" in on else "0.0"), t, "color_jitter prob")
    t = sub_once(r"(horizontal_flip:\s*\n\s*prob:\s*)\S+", r"\g<1>" + ("0.5" if "flip" in on else "0.0"), t, "flip prob")
    t = sub_once(r"(contrast_decrease:\s*\n\s*prob:\s*)\S+(\s*\n\s*factor:\s*)\S+",
                 rf"\g<1>0.5\g<2>{CONTRAST}" if "contrast" in on else r"\g<1>0.0\g<2>0.8", t, "contrast_decrease block")
    t = sub_once(r"(blur_custom:\s*\n\s*prob:\s*)\S+(\s*\n\s*sigma_lo:\s*\S+\s*\n\s*sigma_hi:\s*)\S+",
                 rf"\g<1>0.5\g<2>{blur}" if "blur" in on else r"\g<1>0.0\g<2>1.2", t, "blur_custom block")
    # checks: crop 0.08, brightness strength 0.8 and no other colour jitter, every other operation off
    assert value(t, "rrc", "crop_min_scale") == 0.08, "crop_min_scale is not 0.08"
    assert value(t, "color_jitter", "brightness") == BRIGHT, "brightness strength is not 0.8"
    for k in ("contrast", "saturation", "hue"):
        assert value(t, "color_jitter", k) == 0.0, f"color_jitter {k} is not 0"
    for blk in ("grayscale", "gaussian_blur", "solarization", "equalization", "speckle_noise", "motion_blur", "pixelate",
                "jpeg_compression"):
        assert value(t, blk, "prob") == 0.0, f"{blk} is switched on in p_stem.yaml"
    on_now = {"bright": value(t, "color_jitter", "prob") > 0, "flip": value(t, "horizontal_flip", "prob") > 0,
              "contrast": value(t, "contrast_decrease", "prob") > 0, "blur": value(t, "blur_custom", "prob") > 0}
    assert {k for k, v in on_now.items() if v} == set(on), f"switched on {on_now}, wanted {on}"
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--pairs", nargs="+", default=DEFAULT, choices=PAIRS)
    ap.add_argument("--blur", type=float, default=None, help="blur sigma max for the pairs with blur (e.g. 0.6 or 1.2)")
    a = ap.parse_args()
    if any("blur" in p for p in a.pairs) and a.blur is None:
        raise SystemExit("pairs with blur need --blur (the chosen sigma max)")
    mans = {f: [r["run_id"] for r in csv.DictReader(open(f))] for f in sorted(Path("study").glob("manifest_part*.csv"))}
    planned = {i for ids in mans.values() for i in ids}
    stem_src = (OCT / "augmentations" / "p_stem.yaml").read_text()
    files, rows = {}, []
    for pair in a.pairs:
        on = sorted(pair.split("+"), key=ORDER.index)
        pid = "croponly_" + "_".join(part_id(t, a.blur) for t in on)
        aug_file = f"p_{pid}.yaml"
        files[OCT / "augmentations" / aug_file] = aug_text(stem_src, on, a.blur)
        lv = {}
        if "bright" in on: lv["brightness"] = {"s": BRIGHT}
        if "contrast" in on: lv["contrast"] = {"factor": CONTRAST}
        if "blur" in on: lv["blur"] = {"sigma_max": a.blur}
        if "flip" in on: lv["flip"] = {"prob": 0.5}
        pool_on = [p for p in POOL if p in on]
        for seed in a.seeds:
            for obj in OBJS:
                run_id = f"{obj}-resnet50-octmnist-{pid}" + ("" if seed == 42 else f"-s{seed}")
                cfg = f"{obj}_rn50_p_{pid}" + ("" if seed == 42 else f"_s{seed}")
                r = (OCT / f"{obj}_rn50_p_stem.yaml").read_text()
                r = sub_once(r"(?m)^(\s*-\s*augmentations:\s*)\S+\s*$", rf"\g<1>{aug_file}", r, "augmentations")
                r = sub_once(r'(?m)^name:\s*".*"\s*$', f'name: "{run_id}"', r, "name")
                r = sub_once(r"(?m)^seed:\s*\d+\s*$", f"seed: {seed}", r, "seed")
                files[OCT / f"{cfg}.yaml"] = r
                rows.append(dict(run_id=run_id, part=12, objective=obj, seed=seed, policy_id=pid,
                                 bitmask="".join("1" if p in on else "0" for p in POOL), augs_on="+".join(pool_on),
                                 n_augs=len(pool_on), prob=0.5, levels=json.dumps(lv), crop_min_scale=0.08,
                                 aug_yaml=aug_file, run_yaml=f"{cfg}.yaml", launch_cfg=cfg, priority=1, kind="croponly_pair"))
    new_ids = [r["run_id"] for r in rows]
    same = [f for f, ids in mans.items() if sorted(ids) == sorted(new_ids)]
    if same:
        man = same[0]
    else:
        dup = sorted(i for i in new_ids if i in planned)
        if dup:
            raise SystemExit(f"run ids already planned in another manifest: {dup}")
        man = next(Path(f"study/manifest_part12{s}.csv") for s in [""] + list("bcdefghij")
                   if not Path(f"study/manifest_part12{s}.csv").exists())
    tag = man.stem[len("manifest_part12"):]
    for r in rows:
        print(f"{r['run_id']:58s} seed {r['seed']}  {r['levels']}")
    print(f"\n{len(rows)} runs, {len(files)} new files -> {man}, study/launch_part12{tag}.sh")
    if a.dry_run:
        print("dry run: nothing written"); return
    for p, t in files.items():
        if p.exists() and p.read_text() != t:
            raise SystemExit(f"{p} exists with different content; not overwriting")
    for p, t in files.items():
        p.write_text(t)
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    sh = Path(f"study/launch_part12{tag}.sh")
    sh.write_text(LAUNCH.replace("@MANIFEST@", str(man)).replace("@LAUNCHER@", sh.name))
    sh.chmod(0o755)
    print(f"wrote {man}, {sh} and the yamls")


LAUNCH = r'''#!/usr/bin/env bash
# Part 12 launcher (crop-only pairs, @MANIFEST@). Several copies can run at once: each run is claimed with an atomic
# mkdir in study/claims/, and a run counts as done only when its epoch-99 checkpoint exists.
#   CUDA_VISIBLE_DEVICES=0 bash study/@LAUNCHER@
#   CUDA_VISIBLE_DEVICES=0 bash study/@LAUNCHER@ '^ncl-'    # optional regex filter on run ids
# If a run was interrupted, remove its claim to retry it:  rmdir study/claims/<run_id>
cd "$(dirname "$0")/.." || exit 1
DONE=study/done_part12.txt; TIMING=study/timing_part12.csv; CLAIMS=study/claims
mkdir -p "$CLAIMS"; touch "$DONE"; [ -f "$TIMING" ] || echo "run_id,seconds,gpu_hours,gpu" > "$TIMING"
PAT="${1:-.}"
python3 - <<'PY' > /tmp/part12_queue_$$.txt
import csv
for r in csv.DictReader(open("@MANIFEST@")):
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
    echo "NOT FINISHED $RUN (no epoch-99 checkpoint); claim kept, remove it to retry" | tee -a study/failed_part12.txt
  fi
done < /tmp/part12_queue_$$.txt
rm -f /tmp/part12_queue_$$.txt
echo "launcher finished"
'''

if __name__ == "__main__":
    main()
