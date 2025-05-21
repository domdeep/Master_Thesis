# 5-fold cross-validation with densenet121 and the best hyperparameters
import os, time, json, random, numpy as np
from PIL import Image
import joblib

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall, AUROC
from torch.nn.functional import softmax


dataset_root = r"C:\Users\Xuxu\Desktop\CCMT Dataset"
index_dir = r"C:\Users\Xuxu\Desktop\Master Thesis\OptunaDensenetFull"
optuna_pkl = r"C:/Users/Xuxu/Desktop/Master Thesis/OptunaDensenetFull/new_densenet_study.pkl"
save_dir = r"C:\Users\Xuxu\Desktop\Master Thesis\BestHyperDensenet121Full"
num_folds = 5


os.makedirs(save_dir, exist_ok=True)
torch.set_float32_matmul_precision('medium')

# reproducibility setup
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

# image transformations
training_transformations = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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

# model definition
class DenseNet121Lightning(pl.LightningModule):
    def __init__(self, num_classes, class_weights, hparams):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights", "hparams"])

        self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        # fine-tune all layers
        for param in self.model.features.parameters():
            param.requires_grad = True

        in_features = self.model.classifier.in_features
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

        self.class_weights = class_weights
        self.hparams_dict = hparams
        
        self.loss_fn_train = None
        self.loss_fn_val = nn.CrossEntropyLoss() 

        self.metrics = {
            "f1": F1Score(task="multiclass", num_classes=num_classes, average="macro"),
            "precision": Precision(task="multiclass", num_classes=num_classes, average="macro"),
            "recall": Recall(task="multiclass", num_classes=num_classes, average="macro"),
        }

    def setup(self, stage=None):
        device = self.device
        self.model = self.model.to(device)
        self.loss_fn_train = nn.CrossEntropyLoss(weight=self.class_weights.to(device)) 
        self.loss_fn_val = nn.CrossEntropyLoss()  

        for metric in self.metrics.values():
            metric.to(device)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        x, y = batch
        logits = self(x)
        
        loss_fn = self.loss_fn_train if stage == "train" else self.loss_fn_val
        loss = loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        for name, metric in self.metrics.items():
            value = metric(probs if name == "auc" else preds, y)
            self.log(f"{stage}_{name}", value, on_step=False, on_epoch=True)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

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
    set_seed(42)  
    start_time = time.time()
    
    # load and inspect the best hyperparameters at the beginning
    study = joblib.load(optuna_pkl)
    best_hparams = study.best_trial.params
    best_hparams["lr"] = best_hparams.pop("learning_rate")
    
    # print best hyperparameters for early inspection
    print("Best Hyperparameters Loaded:")
    for key, value in best_hparams.items():
        print(f"  {key}: {value}")
        

    # load dataset split and class info
    fold_results = {}
    fold_data = []

    with open(os.path.join(index_dir, "class_to_idx.json")) as f:
        class_to_idx = json.load(f)
    train_indices = np.load(os.path.join(index_dir, "train_indices.npy"))
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


    # initialize stratified k-fold 
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    val_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_indices, labels[train_indices])):
        print(f"\nStarting training for Fold {fold + 1}/{num_folds}")
        fold_dir = os.path.join(save_dir, f"fold_{fold}", "version_0")
        os.makedirs(fold_dir, exist_ok=True)

        train_set = CustomImageDataset([image_paths[train_indices[i]] for i in train_idx],
                                       [labels[train_indices[i]] for i in train_idx],
                                       transform=training_transformations)
        val_set = CustomImageDataset([image_paths[train_indices[i]] for i in val_idx],
                                     [labels[train_indices[i]] for i in val_idx],
                                     transform=validation_test_transformations)

        train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2, persistent_workers=True, worker_init_fn=worker_init_fn,pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2, persistent_workers=True, worker_init_fn=worker_init_fn,pin_memory=True)

        model = DenseNet121Lightning(len(class_to_idx), class_weights, best_hparams)

        logger = CSVLogger(save_dir=os.path.join(save_dir, f"fold_{fold}"), name="", version="version_0")

        trainer = pl.Trainer(
            max_epochs=25,
            accelerator="gpu",
            devices=1,
            logger=logger,
            enable_checkpointing=False,
            callbacks=[],
            log_every_n_steps=10
        )

        trainer.fit(model, train_loader, val_loader)
        val_f1 = trainer.callback_metrics["val_f1"].item()
        val_f1s.append(val_f1)

        fold_results[f"fold_{fold + 1}"] = {
            "val_f1": val_f1,
            "train_loss": trainer.callback_metrics["train_loss"].item(),
            "train_precision": trainer.callback_metrics["train_precision"].item(),
            "train_recall": trainer.callback_metrics["train_recall"].item(),
        }

        # save indices used in this fold
        fold_data.append((train_idx.tolist(), val_idx.tolist()))

        # save all_preds, all_targets, all_probs
        all_preds, all_targets = [], []

        model.eval()
        model.freeze()
        model.to("cuda")  

        with torch.no_grad():
            for x, y in val_loader:
                x = x.cuda()
                y = y.cuda()
                logits = model(x)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        np.save(os.path.join(fold_dir, "all_preds.npy"), np.array(all_preds))
        np.save(os.path.join(fold_dir, "all_targets.npy"), np.array(all_targets))

        print(f"\nFold {fold + 1}/{num_folds} training complete.")

    # save fold results
    with open(os.path.join(save_dir, "fold_results.json"), "w") as f:
        json.dump(fold_results, f, indent=4)

    # save fold indices (train/val splits)
    np.save(os.path.join(save_dir, "fold_indices.npy"), np.array(fold_data, dtype=object))

    print(f"\nAll {num_folds} folds complete. Total time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
