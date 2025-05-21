import torchvision
import os
import time
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from lightly.loss import NTXentLoss
import math

split_dir = r"C:/Users/Xuxu/Desktop/Master Thesis/OptunaDensenetFull"
dataset_root = r"C:/Users/Xuxu/Desktop/CCMT Dataset"
save_dir = r"C:/Users/Xuxu/Desktop/Master Thesis/SimCLRPretrainImagenet"
batch_size = 256
num_workers = 2
max_epochs = 100
lr = 0.0003
weight_decay = 0.0001
warmup_epochs = 10

torch.set_float32_matmul_precision("medium")
os.makedirs(save_dir, exist_ok=True)

# reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id):
    seed = 42 + worker_id
    np.random.seed(seed)
    random.seed(seed)

# simclr augmentations
ssl_pretrain_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class SimCLRTransformWrapper:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)

# dataset class
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, preload=False):
        self.image_paths = []
        self.preload = preload
        self.data = []

        for crop in os.listdir(root_dir):
            crop_dir = os.path.join(root_dir, crop)
            if not os.path.isdir(crop_dir):
                continue
            for disease in os.listdir(crop_dir):
                disease_dir = os.path.join(crop_dir, disease)
                if not os.path.isdir(disease_dir):
                    continue
                for img in os.listdir(disease_dir):
                    if img.lower().endswith(('jpg', 'jpeg', 'png')):
                        path = os.path.join(disease_dir, img)
                        self.image_paths.append(path)
                        if preload:
                            try:
                                image = Image.open(path).convert("RGB")
                                self.data.append(image.copy())
                            except Exception as e:
                                print(f"[Warning] Could not load {path}: {e}")
                                self.data.append(None)

    def __getitem__(self, i):
        if self.preload and self.data[i] is not None:
            image = self.data[i]
        else:
            image = Image.open(self.image_paths[i]).convert("RGB")
        return image.copy(), 0

    def __len__(self):
        return len(self.image_paths)

class SubsetWithTransform(Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, i):
        image, _ = self.base_dataset[self.indices[i]]
        return self.transform(image), 0

    def __len__(self):
        return len(self.indices)

# simclr lightning module
class SimCLR(pl.LightningModule):
    def __init__(self, lr=lr, weight_decay=weight_decay, max_epochs=max_epochs, projection_dim=128):
        super().__init__()
        self.save_hyperparameters()

        base = models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1)
        )

        self.projection_head = nn.Sequential(
            nn.Linear(768, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim)
        )

        self.criterion = NTXentLoss(temperature=0.1)

        # sanity check
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out = self.encoder(dummy)
            print(f"[Sanity Check] Encoder output shape: {out.shape}")

    def forward(self, x):
        x = self.encoder(x)
        x = self.projection_head(x)
        return nn.functional.normalize(x, dim=1)

    def training_step(self, batch, batch_idx):
        (x0, x1), _ = batch
        z0 = self(x0)
        z1 = self(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        def cosine_warmup_schedule(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / warmup_epochs
            else:
                progress = float(epoch - warmup_epochs) / float(max_epochs - warmup_epochs)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_warmup_schedule)

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

# main training loop
def main():
    set_seed(42)
    start = time.time()

    train_indices = np.load(os.path.join(split_dir, "train_indices.npy"))
    val_indices = np.load(os.path.join(split_dir, "val_indices.npy"))
    pretrain_indices = np.concatenate([train_indices, val_indices])
    print(f"[INFO] Pretraining on {len(pretrain_indices)} images.")

    base_dataset = CustomImageDataset(dataset_root, preload=True)
    simclr_transform = SimCLRTransformWrapper(ssl_pretrain_transform)

    train_loader = DataLoader(
        SubsetWithTransform(base_dataset, pretrain_indices, simclr_transform),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=worker_init_fn
    )

    model = SimCLR()
    logger = CSVLogger(save_dir=save_dir, name="SimCLRPretrain", version="version_0")

    ckpt_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    resume_path = os.path.join(ckpt_dir, "last.ckpt")

    if os.path.exists(resume_path):
        print(f"[RESUME] Resuming from checkpoint at {resume_path}")
    else:
        print("[INFO] No previous checkpoint found. Starting from scratch.")

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="last",
        save_top_k=1,
        save_last=True,
        monitor="train_loss_epoch",
        mode="min",
        verbose=True,
        every_n_epochs=1
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        logger=logger,
        callbacks=[checkpoint_cb],
        log_every_n_steps=1,
        enable_progress_bar=True,
        default_root_dir=save_dir
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        ckpt_path=resume_path if os.path.exists(resume_path) else None
    )

    end_time = time.time()
    minutes, seconds = divmod(int(end_time - start), 60)

    # save encoder
    if os.path.exists(checkpoint_cb.best_model_path):
        best_model = SimCLR.load_from_checkpoint(checkpoint_cb.best_model_path)
        encoder_path = os.path.join(save_dir, "simclr_encoder.pth")
        torch.save({
            "features": best_model.encoder[0].state_dict(),
        }, encoder_path)
        print(f"encoder saved to: {encoder_path}")
        print(f"best checkpoint: {checkpoint_cb.best_model_path}")

if __name__ == "__main__":
    main()
