import os, time, json, random, numpy as np
from PIL import Image
import joblib

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall
from torch.nn.functional import softmax

# configuration and hyperparameters
dataset_root = r"C:\Users\Xuxu\Desktop\CCMT Dataset"
index_dir = r"C:\Users\Xuxu\Desktop\Master Thesis\OptunaDensenetFull"
optuna_pkl = r"C:/Users/Xuxu/Desktop/Master Thesis/OptunaConvNeXtFull/new_convnext_study.pkl"
save_dir = r"C:\Users\Xuxu\Desktop\Master Thesis\BYOLBaselineLabelEfficiencySeed44"

os.makedirs(save_dir, exist_ok=True)
torch.set_float32_matmul_precision('medium')

# reproducibility setup
def set_seed(seed=44):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id):
    seed = 44 + worker_id
    np.random.seed(seed)
    random.seed(seed)

# image transformations for training (with augmentation)
training_transformations = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# image transformations for validation (no augmentation)
validation_test_transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# model definition using convnext-tiny
class ConvNextTinyLightning(pl.LightningModule):
    def __init__(self, num_classes, class_weights, hparams, byol_ckpt_path=None):
        super().__init__()
        self.save_hyperparameters()
        self.hparams_dict = hparams
        self.class_weights = class_weights
        self.num_classes = num_classes

        self.model = models.convnext_tiny(weights=None)

        # load byol pretrained weights
        if byol_ckpt_path is not None and os.path.exists(byol_ckpt_path):
            print(f"\n>> Loading BYOL pretrained weights from {byol_ckpt_path}")
            state_dict = torch.load(byol_ckpt_path, map_location="cpu")
            self.model.features.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"BYOL checkpoint not found at {byol_ckpt_path}")

        # unfreeze encoder for fine-tuning
        for param in self.model.features.parameters():
            param.requires_grad = True

        # build classifier head with configurable fc layers
        in_features = self.model.classifier[2].in_features
        fc_hidden_dim = hparams["fc_hidden_dim"]
        dropout = hparams["dropout"]
        fc_layers = hparams.get("fc_layers", 1)

        layers = [nn.Flatten(start_dim=1)]

        if fc_layers == 1:
            layers.append(nn.Linear(in_features, num_classes))
        else:
            layers.append(nn.Linear(in_features, fc_hidden_dim))
            layers.append(nn.BatchNorm1d(fc_hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))

            for _ in range(fc_layers - 2):
                layers.append(nn.Linear(fc_hidden_dim, fc_hidden_dim))
                layers.append(nn.BatchNorm1d(fc_hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout))

            layers.append(nn.Linear(fc_hidden_dim, num_classes))

        self.model.classifier = nn.Sequential(*layers)

    def setup(self, stage=None):
        device = self.device
        self.loss_fn_train = nn.CrossEntropyLoss(weight=self.class_weights.to(device))
        self.loss_fn_test = nn.CrossEntropyLoss()

        self.metrics = {
            "f1": F1Score(task="multiclass", num_classes=self.num_classes, average="macro").to(device),
            "precision": Precision(task="multiclass", num_classes=self.num_classes, average="macro").to(device),
            "recall": Recall(task="multiclass", num_classes=self.num_classes, average="macro").to(device),
            "acc": Accuracy(task="multiclass", num_classes=self.num_classes).to(device),
        }

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        x, y = batch
        logits = self(x)

        loss_fn = self.loss_fn_train if stage == "train" else self.loss_fn_test
        loss = loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        for name, metric in self.metrics.items():
            value = metric(preds, y)
            self.log(f"{stage}_{name}", value, on_epoch=True, on_step=False, prog_bar=True)

        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = {
            "Adam": optim.Adam,
            "AdamW": optim.AdamW,
            "RMSprop": lambda p, **kw: optim.RMSprop(p, momentum=0.9, **kw)
        }[self.hparams_dict["optimizer"]](
            params,
            lr=self.hparams_dict["lr"],
            weight_decay=self.hparams_dict["weight_decay"]
        )
        return optimizer

# training and evaluation pipeline
def main():
    set_seed(44)
    start_time = time.time()

    # load best hyperparameters from optuna
    study = joblib.load(optuna_pkl)
    best_hparams = study.best_trial.params
    best_hparams["lr"] = best_hparams.pop("learning_rate")
    print("Best Hyperparameters:", best_hparams)

    # load label to index mapping
    with open(os.path.join(index_dir, "class_to_idx.json")) as f:
        class_to_idx = json.load(f)

    # load dataset split indices
    train_indices = np.load(os.path.join(index_dir, "train_indices_seed44.npy"))
    val_indices = np.load(os.path.join(index_dir, "val_indices_seed44.npy"))
    test_indices = np.load(os.path.join(index_dir, "test_indices.npy"))
    class_weights = torch.load(os.path.join(index_dir, "class_weights.pt"))

    image_paths, labels = [], []
    for crop in sorted(os.listdir(dataset_root)):
        for disease in sorted(os.listdir(os.path.join(dataset_root, crop))):
            class_name = f"{crop}_{disease}"
            class_idx = class_to_idx[class_name]
            img_dir = os.path.join(dataset_root, crop, disease)
            for img_file in os.listdir(img_dir):
                if img_file.lower().endswith(("jpg", "jpeg", "png")):
                    image_paths.append(os.path.join(img_dir, img_file))
                    labels.append(class_idx)
    labels = np.array(labels)

    # combine training and validation for full training set
    combined_train_indices = np.concatenate([train_indices, val_indices])

    test_set = CustomImageDataset(
        [image_paths[i] for i in test_indices],
        [labels[i] for i in test_indices],
        transform=validation_test_transformations
    )
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=2, persistent_workers=True, worker_init_fn=worker_init_fn)

    # define the different fractions (%) of the combined train+val set to use for training
    label_percentages = [5, 10, 20, 40, 60, 80, 100]

    for percent in label_percentages:

        set_seed(44)  
        
        num_samples = int(len(combined_train_indices) * (percent / 100))
        selected_indices = np.random.choice(combined_train_indices, num_samples, replace=False)

        train_set = CustomImageDataset(
            [image_paths[i] for i in selected_indices],
            [labels[i] for i in selected_indices],
            transform=training_transformations
        )
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2, persistent_workers=True, worker_init_fn=worker_init_fn)
        
        # load pretrained encoder from byol
        byol_ckpt_path = r"C:/Users/Xuxu/Desktop/Master Thesis/BYOLPretrainVer3/byol_encoder.pt"
        model = ConvNextTinyLightning(len(class_to_idx), class_weights, best_hparams, byol_ckpt_path=byol_ckpt_path)
    
        log_name = f"label_efficiency_{percent}percent"
        log_version = "version_0"
        version_dir = os.path.join(save_dir, log_name, log_version)
        pred_dir = os.path.join(version_dir, "predictions")

        os.makedirs(pred_dir, exist_ok=True)

        logger = CSVLogger(
            save_dir=save_dir,
            name=log_name,
            version=log_version
        )

        trainer = pl.Trainer(
            max_epochs=100,
            accelerator="gpu",
            devices=1,
            logger=logger,
            precision="16-mixed",
            enable_checkpointing=False,
            callbacks=[],
            log_every_n_steps=1
        )

        trainer.fit(model, train_loader)
        trainer.test(model, dataloaders=test_loader, verbose=True)

        all_preds, all_targets, all_probs = [], [], []
        model.eval()
        model.freeze()
        model.to("cuda")

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                logits = model(x)
                probs = softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        metrics_to_save = {k: (v.item() if torch.is_tensor(v) else v) for k, v in trainer.callback_metrics.items()}

        with open(os.path.join(version_dir, "test_metrics.json"), "w") as f:
            json.dump(metrics_to_save, f, indent=2)

        np.save(os.path.join(version_dir, "all_preds.npy"), np.array(all_preds))
        np.save(os.path.join(version_dir, "all_targets.npy"), np.array(all_targets))
        np.save(os.path.join(version_dir, "all_probs.npy"), np.array(all_probs))
        
        torch.save(model.state_dict(), os.path.join(version_dir, "model.pth"))

    print(f"\nLabel Efficiency Testing complete. Total time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
