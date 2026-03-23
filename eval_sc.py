import argparse, json, torch, sys
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",       required=True)
parser.add_argument("--dataset",    default="cifar100")
parser.add_argument("--data_path",  default="./data")
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--num_workers",default=4,   type=int)
parser.add_argument("--ncl",        default="auto")
args = parser.parse_args()

apply_relu = ("ncl" in args.ckpt.lower()) if args.ncl.lower() == "auto" else (args.ncl.lower() == "true")
print(f"NCL mode (apply ReLU): {apply_relu}")

sys.path.insert(0, ".")
from solo.methods import METHODS
from solo.args.pretrain import parse_cfg

# Load args.json from same folder as checkpoint
ckpt_dir = Path(args.ckpt).parent
args_json = ckpt_dir / "args.json"
print(f"Loading config from: {args_json}")
with open(args_json) as f:
    saved_args = json.load(f)

cfg = OmegaConf.create(saved_args)
OmegaConf.set_struct(cfg, False)
cfg = parse_cfg(cfg)

model = METHODS[cfg.method](cfg)
ckpt = torch.load(args.ckpt, map_location="cpu")
model.load_state_dict(ckpt["state_dict"])
model = model.cuda().eval()
print(f"Loaded: {args.ckpt}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
train_dataset = datasets.CIFAR100(args.data_path, train=True,  download=True, transform=transform)
val_dataset   = datasets.CIFAR100(args.data_path, train=False, download=True, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
val_loader    = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)

def extract(loader):
    all_feats, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.cuda())
            z = F.relu(out["z"]) if apply_relu else out["z"]
            all_feats.append(z.cpu())
            all_labels.append(y)
    return torch.cat(all_feats), torch.cat(all_labels)

print("Extracting train features...")
train_feats, train_labels = extract(train_loader)
print("Extracting val features...")
val_feats, val_labels = extract(val_loader)
print(f"Train features: {train_feats.shape}, Val features: {val_feats.shape}")

def sparsity(f, eps=1e-5):
    sp = (f.abs() < eps).float().mean(dim=1) * 100
    print(f"Sparsity: {sp.mean():.2f}%")

def ccr(features, labels, split, eps=1e-5):
    # normalize to match training-time semantic_consistency()
    features = F.normalize(features, dim=1)
    N, D = features.shape
    n_classes = labels.max().item() + 1
    scores = []
    for d in range(D):
        mask = features[:, d].abs() > eps
        if mask.sum() == 0:
            continue
        counts = torch.bincount(labels[mask], minlength=n_classes).float()
        scores.append(counts.max().item() / mask.sum().item())
    scores = torch.tensor(scores) * 100
    print(f"==================================================")
    print(f"Semantic Consistency (CCR) [{split}]")
    print(f"  Active dims : {len(scores)} / {D}")
    print(f"  CCR mean    : {scores.mean():.2f}%")
    print(f"  CCR std     : {scores.std():.2f}%")
    print(f"==================================================")

print("\n--- Train set ---")
sparsity(train_feats)
ccr(train_feats, train_labels, "train")

print("\n--- Val set ---")
sparsity(val_feats)
ccr(val_feats, val_labels, "val")
