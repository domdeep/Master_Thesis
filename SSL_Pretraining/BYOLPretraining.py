# byol  pretraining 
# adapted from: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/byol.ipynb
# modified for custom dataset, convnext_tiny backbone

import os
import time
import copy
import random
import math
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as T
from torchvision.transforms import (
    RandomApply, GaussianBlur, RandomGrayscale, ColorJitter,
    RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize
)

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from lightly.models.modules import BYOLProjectionHead, BYOLPredictionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum

# training config
split_dir = r"C:/Users/Xuxu/Desktop/Master Thesis/OptunaDensenetFull"
dataset_root = r"C:/Users/Xuxu/Desktop/CCMT Dataset"
save_dir = r"C:/Users/Xuxu/Desktop/Master Thesis/BYOLPretrainVer3"
batch_size = 256
num_workers = 2
max_epochs = 100
lr = 0.0003
weight_decay = 0.0001

os.makedirs(save_dir, exist_ok=True)
torch.set_float32_matmul_precision("medium")

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

# BYOL-style dual view transform
class BYOLTransformView:
    def __init__(self, input_size=224):
        self.left_transform = T.Compose([
            RandomResizedCrop(input_size, scale=(0.08, 1.0)),
            RandomHorizontalFlip(p=0.5),
            ColorJitter(0.8, 0.8, 0.8, 0.2),
            RandomGrayscale(p=0.2),
            RandomApply([GaussianBlur(23, sigma=(0.1, 2.0))], p=0.1),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.right_transform = T.Compose([
            RandomResizedCrop(input_size, scale=(0.08, 1.0)),
            RandomHorizontalFlip(p=0.5),
            ColorJitter(0.8, 0.8, 0.8, 0.2),
            RandomGrayscale(p=0.2),
            RandomApply([GaussianBlur(23, sigma=(0.1, 2.0))], p=0.1),
            RandomApply([T.RandomSolarize(128)], p=0.2), # solarization only for right view
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    def __call__(self, x):
        return self.left_transform(x), self.right_transform(x)

# Custom dataset returning BYOL dual views
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, indices=None):
        all_paths = []
        for crop_folder in os.listdir(root_dir):
            crop_path = os.path.join(root_dir, crop_folder)
            if not os.path.isdir(crop_path): continue
            for sub in os.listdir(crop_path):
                sub_path = os.path.join(crop_path, sub)
                if not os.path.isdir(sub_path): continue
                for img_name in os.listdir(sub_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        path = os.path.join(sub_path, img_name)
                        try:
                            Image.open(path).verify()
                            all_paths.append(path)
                        except:
                            continue
        self.image_paths = [all_paths[i] for i in indices] if indices is not None else all_paths
        self.transform = BYOLTransformView()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)

# BYOL LightningModule
class BYOLConvNeXtTiny(pl.LightningModule):
    def __init__(self):
        super().__init__()
        base = torchvision.models.convnext_tiny(weights=None)
        self.backbone = nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.projection_head = BYOLProjectionHead(768, 4096, 256)
        self.prediction_head = BYOLPredictionHead(256, 4096, 256)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.prediction_head(self.projection_head(self.backbone(x)))

    def forward_momentum(self, x):
        with torch.no_grad():
            return self.projection_head_momentum(self.backbone_momentum(x)).detach()

    def training_step(self, batch, batch_idx):
        x0, x1 = batch

        # update EMA momentum
        base_momentum = 0.996
        max_e = self.trainer.max_epochs
        epoch = self.current_epoch
        momentum = 1 - (1 - base_momentum) * (math.cos(math.pi * epoch / max_e) + 1) / 2
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)

        # forward pass
        p0, z1 = self.forward(x0), self.forward_momentum(x1)
        p1, z0 = self.forward(x1), self.forward_momentum(x0)

        # BYOL loss = MSE between normalized vectors
        loss = 0.5 * (
            self.criterion(F.normalize(p0, dim=-1), F.normalize(z1.detach(), dim=-1)) +
            self.criterion(F.normalize(p1, dim=-1), F.normalize(z0.detach(), dim=-1))
        )

        if torch.isnan(loss):
            raise ValueError(f"[nan detected] epoch {self.current_epoch} | batch {batch_idx}")
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        def cosine_warmup_schedule(epoch):
            if epoch < 10:
                return float(epoch + 1) / 10
            else:
                progress = float(epoch - 10) / float(max_epochs - 10)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_warmup_schedule)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

# training loop
def main():
    set_seed(42)
    start = time.time()

    print("[info] loading indices...")
    train_idx = np.load(os.path.join(split_dir, "train_indices.npy"))
    val_idx = np.load(os.path.join(split_dir, "val_indices.npy"))
    indices = np.concatenate([train_idx, val_idx])

    dataset = CustomImageDataset(dataset_root, indices=indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True,
                            persistent_workers=True, worker_init_fn=worker_init_fn)

    model = BYOLConvNeXtTiny()
    logger = CSVLogger(save_dir=save_dir, name="BYOLConvNeXt")

    ckpt_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        logger=logger,
        callbacks=[ModelCheckpoint(dirpath=ckpt_dir, filename="last", save_last=True)],
        log_every_n_steps=1,
        default_root_dir=save_dir
    )

    print("[info] starting training...")
    trainer.fit(model, train_dataloaders=dataloader)

    # save encoder
    if os.path.exists(os.path.join(ckpt_dir, "last.ckpt")):
        best_model = BYOLConvNeXtTiny.load_from_checkpoint(os.path.join(ckpt_dir, "last.ckpt"))
        encoder_path = os.path.join(save_dir, "byol_encoder.pth")
        torch.save({
            "features": best_model.backbone[0].state_dict(),
        }, encoder_path)
        print(f"Encoder saved to: {encoder_path}")
        print(f"Best checkpoint: {os.path.join(ckpt_dir, 'last.ckpt')}")

    print(f"[done] Total training time: {int(time.time() - start)} seconds")

if __name__ == "__main__":
    main()
