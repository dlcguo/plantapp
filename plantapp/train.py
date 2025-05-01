import torch, torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from config import cfg
from data.dataset import PlantDataset
from models.resnet_twohead import ResNetTwoHead

# This script trains a two-headed ResNet model for the plantapp project

def main():
    device = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available()
               else "cpu")

    train_tf = T.Compose([
        T.RandomResizedCrop(cfg.img_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(p=0.2),
        T.ColorJitter(0.2, 0.2, 0.1, 0.05),
        T.ToTensor(),
        T.Normalize(cfg.imagenet_mean, cfg.imagenet_std),
    ])

    val_tf = T.Compose([
        T.Resize(int(cfg.img_size * 1.14)),
        T.CenterCrop(cfg.img_size),
        T.ToTensor(),
        T.Normalize(cfg.imagenet_mean, cfg.imagenet_std),
    ])

    train_ds = PlantDataset(cfg.train_split_dir, transform=train_tf)
    val_ds   = PlantDataset(cfg.val_split_dir,   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                            shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers)

    model = ResNetTwoHead(len(train_ds.plant_to_idx), len(train_ds.disease_to_idx)).to(device)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.initial_lr,
                              weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    for epoch in range(1, cfg.epochs + 1):
        # Training
        model.train()
        run_loss = seen = 0
        bar = tqdm(train_loader, desc=f"Ep{epoch:02d}(training)")
        for x,(p,d) in bar:
            x, p, d = x.to(device), p.to(device), d.to(device)
            plogits, dlogits = model(x)
            loss = crit(plogits,p) + crit(dlogits,d)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * x.size(0)
            seen += x.size(0)
            bar.set_postfix(loss=f"{run_loss/seen:.3f}")
        sched.step()

        # Validation
        model.eval()
        cp = cd = tot = 0
        with torch.no_grad():
            for x,(p, d) in val_loader:
                x, p, d = x.to(device), p.to(device), d.to(device)
                plogits, dlogits = model(x)
                cp += (plogits.argmax(1)==p).sum().item()
                cd += (dlogits.argmax(1)==d).sum().item()
                tot += x.size(0)
        print(f"Ep{epoch:02d}  val plant={cp/tot:.3f}  disease={cd/tot:.3f}")

    torch.save(model.state_dict(), cfg.checkpoint_path)
    print("model saved to", cfg.checkpoint_path)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()