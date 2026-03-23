# Reproduction Findings: Non-negative Contrastive Learning (NCL)

**Paper:** Non-negative Contrastive Learning, Wang et al., ICLR 2024
**Dataset:** CIFAR-100, ResNet-18, 200 epochs

---

## 1. Results

All results use `eval_sc_TC.py`, which implements the same CCR logic as the paper's `main_eval.py` (see Section 3 for verification).

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

The paper states (Section 5): *"we run 3 random trials and report their mean and standard deviation."* 
The 8.2%  is the mean over 3 runs. Our single run gets 10.05–10.20%. Should I also run three times? Not sure if this will change much.

---

Please look at:

### Training:
  - scripts/pretrain/cifar/ncl.yaml — training hyperparameters (batch size, lr, epochs, optimizer)
  - solo/methods/simclr.py — where rep_relu is applied to z (line 154 or close)

### Eval:
  - eval_sc_TC.py — your standalone eval script, runs CCR on the test set and prints the result
  - main_eval.py — the paper's eval script 

### CCR/ Semantic Consistency:
  - solo/methods/simclr.py lines 170–195:semantic_consistency() function, computed during training and logged to WandB. They compute CCR at two different points
  - main_eval.py lines 184–218: cluster_acc() function, the offline version used for reporting