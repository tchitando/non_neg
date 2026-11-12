# How the augmentation study code works

Repository root: `/data/tchitando/Non_neg` on the server (mirror on the Mac: `~/Desktop/NWU/Non_neg`). Run every command from the root, in the `ml` conda environment.

## 1. One run, end to end

A run is one combination of **dataset × objective (NCL or SimCLR) × augmentation policy × seed**, for example `ncl-resnet50-octmnist-contrast-s43`.

```
manifest row ──> launcher ──> MedMNIST/main_pretrain.py  (reads the run yaml with Hydra)
                                   │
                                   ├─ pretrain_dataloader.build_transform_pipeline   (the augmentation policy)
                                   ├─ pretrain_dataloader.prepare_datasets            (training images, no labels used)
                                   ├─ solo/methods/simclr.py                          (ResNet-50 + projector + loss)
                                   └─ checkpoint: trained_models/simclr/offline-<id>/<run_id>-offline-<id>-ep=99.ckpt
                                                │
study_tools/eval_run.py   ─────────────────────>  study/results_master.csv  (probe, k-NN, SC, MS, effective dimension)
study_tools/eval_spec.py  ─────────────────────>  study/results_spec.csv    (SC*, purity, MS*, shuffled and untrained controls)
study_tools/make_paper_results.py ─────────────>  paper tables, figures, results_numbers.json
```

## 2. Files and where they are

| Where | File | What it does |
|---|---|---|
| `MedMNIST/` | `main_pretrain.py` | Entry point for every pretraining run (Hydra config, model, data, trainer, checkpoints). |
| | `pretrain_dataloader.py` | Builds the augmentation pipeline from the augmentation yaml; speckle and contrast reduction live here; loads the pretraining dataset. |
| | `medmnistc_transforms.py` | The other pool operations: Gaussian blur range, motion blur, pixelation, JPEG (OCT), and rotation/flip, stain jitter, defocus blur (PathMNIST). Wraps MedMNIST-C, fixing its bugs at 64 px. |
| | `classification_dataloader.py` | Validation loader for the online classifier that is trained during pretraining (monitoring only). |
| | `args_pretrain.py` | Config checks; list of supported datasets and their class counts. |
| | `octmnist_dataset.py`, `pathmnist_dataset.py` | Read `octmnist_64.npz` / `pathmnist_64.npz` (64 px MedMNIST+) and return PIL RGB images. |
| `MedMNIST/scripts/pretrain/octmnist/` | `augmentations/p_<policy>.yaml` | One augmentation policy (stem + which pool operations, at what strength). |
| | `{ncl,simclr}_rn50_p_<policy>[_s43\|_s44].yaml` | One run: architecture, optimiser, dataset, seed, and which augmentation file. |
| `MedMNIST/scripts/pretrain/pathmnist_study/` | same layout | The PathMNIST runs. |
| `solo/methods/` | `simclr.py`, `base.py` | The loss (NCL's non-negative mapping is here) and the backbone/projector/online classifier. |
| `study/` | `manifest_part*.csv`, `launch_part*.sh` | The planned OCT runs and their launchers (Part 1–4). |
| | `claims/`, `done_part*.txt`, `timing_part*.csv`, `failed_part*.txt` | Run bookkeeping (who is running what, what finished, how long it took). |
| | `results_master.csv`, `results_spec.csv` | All scores. Append-only: the latest row per run is the one used. |
| | `levels.json`, `label_test*/` | The chosen operation strengths and the label-preservation tests behind them. |
| `study/path/` | `manifest.csv`, `launch.sh`, `label_test/`, results files | The same for PathMNIST. |
| `study_tools/` | `make_runs.py`, `make_confirm_runs.py`, `make_part4_runs.py`, `make_path_runs.py` | Generate run yamls, manifests and launchers. |
| | `aug_ladders.py`, `path_pool.py` | Strength ladders for each operation (OCT, PathMNIST). |
| | `label_test.py`, `label_test_aug.py`, `label_test_stack.py`, `label_test_pool.py` | Label-preservation tests. |
| | `eval_run.py`, `eval_spec.py` | Score finished runs (validation split only). |
| | `make_paper_results.py` | Tables, figures and every number quoted in the paper. |
| | `gallery.py`, `test_transforms.py`, `smoke_path.py` | Visual and timing checks of the augmentations. |
| | `add_pathmnist.py` | One-off installer that added PathMNIST to the pipeline. |
| `trained_models/simclr/offline-*/` | `*-ep=99.ckpt` | Checkpoints. A run counts as finished only when its epoch-99 checkpoint exists. |
| `paper/` | LaTeX | Paper sections, tables, figures and the OCT analysis. |

## 3. How the augmentations (corruptions) are applied

Each training image is turned into **two views**. `NCropAugmentation` calls the same pipeline twice, so every random choice (crop, whether each operation fires, its random parameters) is drawn **independently for each view**. The pipeline, in this exact order (`build_transform_pipeline`):

| Step | Operation | Setting | Probability |
|---|---|---|---|
| 1 | Random resized crop to 64 px | area in [crop_min_scale, 1], bicubic | always |
| 2 | Colour jitter | brightness factor in [0.2, 1.8] (stem); natural-image policy also contrast, saturation 0.8, hue 0.2 | 0.8 |
| 3 | Grayscale, Gaussian blur σ∈[0.1, 2], solarise, equalise | off, except grayscale 0.2 and blur 0.5 in the natural-image policy | — |
| 4 | Horizontal flip | | 0.5 |
| 5 | Speckle noise (OCT pool) | x ← x + x·ε, ε ~ N(0, std²), drawn per pixel and per RGB channel | 0.5 if in the policy |
| 6 | Contrast reduction (both pools) | x ← μ + f·(x − μ), μ = image mean | 0.5 if in the policy |
| 7 | Rotation + vertical flip (PathMNIST) | random multiple of 90°, then vertical flip 0.5 | 0.5 if in the policy |
| 8 | Stain jitter (PathMNIST) | haematoxylin/eosin/DAB concentrations c ← c·a + b, a ~ U[1−σ, 1+σ], b ~ U[−σ, σ] | 0.5 if in the policy |
| 9 | Gaussian blur (OCT pool) | σ ~ U[0.1, σ_max] | 0.5 if in the policy |
| 10 | Defocus blur (PathMNIST) | MedMNIST-C disk kernel, radius r | 0.5 if in the policy |
| 11 | Motion blur (OCT pool) | ImageMagick, radius 5, σ 2, angle ±15° | 0.5 if in the policy |
| 12 | Pixelation (both pools) | box down-sampling by a factor, then up-sampling | 0.5 if in the policy |
| 13 | JPEG (both pools) | quality q | 0.5 if in the policy |
| 14 | ToTensor, Normalise | dataset mean and std | always |

Steps 1–4 are the **stem**, shared by every policy. A **policy** is the stem plus a subset of the pool operations; the yaml block of an operation that is not in the policy has `prob: 0.0`, and the code skips it. Every operation has **one fixed strength** (from the label-preservation test, stored in `study/levels.json` or `study/path/label_test/levels.json`) and probability 0.5 in every policy, so policies differ only in which operations they contain. Strength runs (crop 0.2/0.5, contrast factor 0.6/0.9) change exactly one value in a copied yaml.

**MedMNIST-C**: the operations follow MedMNIST-C's definitions (and use its motion-blur and defocus kernels), but the strengths are not MedMNIST-C's severity levels. Its ladders for motion blur, pixelation and JPEG are set for 224 px images and fail the label test at every level at 64 px, so the strongest passing level of a milder ladder is used. `medmnistc_transforms.py` also works around three bugs in medmnistc 0.1.0 (listed at the top of the file).

**Nuisance sharing**: the dataloader can correlate speckle and contrast between the two views (`sharing` r). It is 0 in every study run.

## 4. Training recipe (identical for every run)

ResNet-50 from random initialisation with the standard ImageNet stem (7×7 convolution, max-pool; the CIFAR stem is only used for CIFAR), projector 2048 → 2048 → 256 with a ReLU between the layers, InfoNCE at temperature 0.2 over the two views, LARS (lr 0.4, weight decay 1e-4), batch 256, 10 warm-up epochs then cosine to 100 epochs, mixed precision, one GPU. **NCL** differs from SimCLR in one place (`solo/methods/simclr.py`): the projector output is normalised, passed through `rep_relu` (ReLU in the forward pass, GELU gradient in the backward pass) and normalised again before the loss. The online linear classifier is trained on detached features and does not affect the encoder.

## 5. Scoring (validation split only)

`eval_run.py` freezes the checkpoint and extracts the backbone feature h (2048-d) and projector output z (256-d) for the training and validation images. It reports the linear probe on h (LARS, 50 epochs, validation balanced accuracy, the primary score), the probe on z, k-NN (k = 5, 20, 100, cosine) on h, SC, alive units and effective dimension.
`eval_spec.py` (version s2) reads z as the loss sees it (normalised; for NCL also the non-negative mapping), and reports SC* with its shuffled-label value, top-100 class purity on z and h (chance-corrected), and MS on a fixed class-balanced subset of 500 images per class with DINOv2 ViT-B/14, rescaled by its no-preference floor. `--random-init 3` adds untrained-network controls. Both scripts take `--dataset octmnist|pathmnist`.

## 6. Bookkeeping rules

- A launcher claims a run with an atomic `mkdir study/claims/<run_id>` before starting it, so several launchers can share one queue without running anything twice. To retry an interrupted run: `rmdir study/claims/<run_id>`.
- A run is done only when its `ep=99` checkpoint exists.
- Result tables are append-only; nothing is edited by hand. `--force` appends a new row, and the latest row per run wins.
- The test split is never read by any of these scripts.
