# Non-negative Contrastive Learning — Reproduction

Reproduction of [Non-negative Contrastive Learning](https://arxiv.org/pdf/2403.12459) (Wang et al., ICLR 2024) on CIFAR-100.

Original repo: https://github.com/PKU-ML/non_neg

---

## Installation

```bash
git clone https://github.com/tchitando/non_neg
cd non_neg
pip install -e .
```

Requires Python 3.8, PyTorch, and CUDA.

---

## Pretrained Checkpoints

Download from Google Drive and place in `trained_models/`:

https://drive.google.com/drive/folders/1zSGE2_sVo0FcQ79MXXAkoac3xvQ_qkKB?usp=drive_link

| Model | Path |
|---|---|
| SimCLR baseline | `trained_models/simclr/10/simclr-resnet18-cifar100-10-ep=199.ckpt` |
| NCL (rep_relu) | `trained_models/simclr/11/simclr-resnet18-cifar100-ncl-11-ep=199.ckpt` |

---

## Evaluation

### Standalone script 

```bash
# NCL
python eval_sc_TC.py \
  --ckpt trained_models/simclr/11/simclr-resnet18-cifar100-ncl-11-ep=199.ckpt \
  --ncl true

# SimCLR baseline
python eval_sc_TC.py \
  --ckpt trained_models/simclr/10/simclr-resnet18-cifar100-10-ep=199.ckpt \
  --ncl false
```

## Pretraining from scratch

```bash
# SimCLR
python main_pretrain.py --config-path scripts/pretrain/cifar --config-name simclr.yaml

# NCL
python main_pretrain.py --config-path scripts/pretrain/cifar --config-name ncl.yaml
```

---

## Results

See [FINDINGS.md](FINDINGS.md) for full results.

