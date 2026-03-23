import math
from torch.optim.lr_scheduler import _LRScheduler

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs,
                 warmup_start_lr=0.0, eta_min=0.0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # linear warmup
            if self.warmup_epochs == 0:
                return self.base_lrs
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                    for base_lr in self.base_lrs]
        # cosine annealing
        progress = (self.last_epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return [self.eta_min + cosine * (base_lr - self.eta_min)
                for base_lr in self.base_lrs]
