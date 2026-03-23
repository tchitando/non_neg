# Reproduction Findings: Non-negative Contrastive Learning (NCL)

**Paper:** Non-negative Contrastive Learning, Wang et al., ICLR 2024
**Dataset:** CIFAR-100, ResNet-18, 200 epochs

---

## 1. Results

All results use `eval_sc_TC.py`, which implements the same CCR logic as the paper's `main_eval.py`. SC is computed over the full 10k validation set with no augmentations.

| Checkpoint | CCR (test) |
|---|---|
| SimCLR — our checkpoint | 1.00% |
| SimCLR — paper's official checkpoint | 1.00% |
| NCL — our checkpoint | 10.05% |
| NCL — paper's official checkpoint | 10.20% |
| NCL — paper reported | 8.2% |

**SimCLR** gives 1.00% similar to their checkpoint (same eval).

**NCL** gives 10.05% on our checkpoint and 10.20% on the paper's official checkpoint (same eval). 


## Thoughts

The paper has two evaluation sections. These give completely different results.

Training-time SC (W&B / TensorBoard logs): Using solo/methods/simclr.py (lines 170–195), SC is computed at batch_idx == 0 each epoch on a single batch of 256 × 2 augmented training samples, then logged automatically. 

Run it with 

```bash
python read_sc_logs.py
```

Reading values directly from TensorBoard and getting a mean: Mean (all epochs): 11.72% for NCL and 2.83% for SimCLR.

Maybe the 8.2% was their best result?

---

Please look at:

### Training:
  - scripts/pretrain/cifar/ncl.yaml — training hyperparameters (batch size, lr, epochs, optimizer)
  - solo/methods/simclr.py — where rep_relu is applied to z (line 154 or close)

### Eval:
  - eval_sc_TC.py — the standalone eval script, runs CCR on the test set and prints the result
  - main_eval.py — the paper's eval script 

### CCR/ Semantic Consistency:
  - solo/methods/simclr.py lines 170–195:semantic_consistency() function, computed during training and logged to WandB. They compute CCR at two different points
  - main_eval.py lines 184–218: cluster_acc() function, the offline version used for reporting

---

### Differences between the NCL main_eval and my compute_semantic_consistency
( I used this one for BYI because the BYI repo had no implementation of SC)

Features extracted  -> h['z'] — takes projector output, but wrong projector dims (512→16384→2048 instead of 512→2048→256).
Should use the full 10,000 samples of the test set instead of the training set, 50 random batches of 512.
