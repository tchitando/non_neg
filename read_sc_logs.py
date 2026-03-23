from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
# version_10 = SimCLR, version_18 = NCL (best full 200-epoch runs)
for version, name in [('version_10', 'SimCLR'), ('version_18', 'NCL')]:
    ea = EventAccumulator(f'lightning_logs/{version}')
    ea.Reload()
    events = ea.Scalars('semantic_consistency')
    values = [e.value * 100 for e in events]
    t = torch.tensor(values)
    print(f"\n{name} ({version})")
    
    print(f"  Mean (all epochs): {t.mean():.2f}%")
    
