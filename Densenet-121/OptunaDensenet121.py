# densenet121 with optuna hyperparameter tuning

import os
import random
import numpy as np
import json
from PIL import Image
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall, MulticlassF1Score
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import optuna
import joblib
from optuna.exceptions import TrialPruned

# reproducibility setup
torch.set_float32_matmul_precision('medium')

# set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ensure each data loader worker has a different seed
def worker_init_fn(worker_id):
    seed = 42 + worker_id
    np.random.seed(seed)
    random.seed(seed)

# data augmentations for training 
training_transformations = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# data augmentations for validation 
validation_test_transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# custom dataset to load images and labels
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# prepare dataset with stratified train/val/test split and optional saving
def setup_ssl_dataset(dataset_root, split_save_path=None):
    image_paths, labels, class_to_idx = [], [], {}
    image_paths_by_class = defaultdict(list)

    for crop in sorted(os.listdir(dataset_root)):
        crop_path = os.path.join(dataset_root, crop)
        if not os.path.isdir(crop_path):
            continue
        for disease in sorted(os.listdir(crop_path)):
            disease_path = os.path.join(crop_path, disease)
            if not os.path.isdir(disease_path):
                continue
            class_name = f"{crop}_{disease}"
            if class_name not in class_to_idx:
                class_to_idx[class_name] = len(class_to_idx)
            class_idx = class_to_idx[class_name]
            for img_file in os.listdir(disease_path):
                if img_file.lower().endswith(('jpg', 'jpeg', 'png')):
                    img_path = os.path.join(disease_path, img_file)
                    image_paths_by_class[class_idx].append(img_path)

    for class_idx, paths in image_paths_by_class.items():
        image_paths.extend(paths)
        labels.extend([class_idx] * len(paths))

    all_indices = np.arange(len(labels))
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        all_indices, labels, test_size=0.2, stratify=labels, random_state=42)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=42)

    if split_save_path:
        os.makedirs(split_save_path, exist_ok=True)
        np.save(os.path.join(split_save_path, "train_indices.npy"), train_idx)
        np.save(os.path.join(split_save_path, "val_indices.npy"), val_idx)
        np.save(os.path.join(split_save_path, "test_indices.npy"), test_idx)
        with open(os.path.join(split_save_path, "class_to_idx.json"), "w") as f:
            json.dump(class_to_idx, f, indent=4)

    print(f"[INFO] Final split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    return (
        CustomImageDataset([image_paths[i] for i in train_idx], [labels[i] for i in train_idx], training_transformations),
        CustomImageDataset([image_paths[i] for i in val_idx], [labels[i] for i in val_idx], validation_test_transformations),
        labels,
        len(set(labels)),
        class_to_idx,
        train_idx, val_idx, test_idx
    )

# densenet121 with configurable classifier head and optimizer
class DenseNet121Lightning(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-4, dropout=0.5, weight_decay=1e-5,
                 fc_hidden_dim=512, optimizer="Adam", class_weights=None, fc_layers=1):
        super().__init__()
        self.save_hyperparameters()

        self.model = models.densenet121(weights='IMAGENET1K_V1')
        for p in self.model.features.parameters():
            p.requires_grad = True

        layers = [nn.Flatten(start_dim=1)]
        in_features = self.model.classifier.in_features

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

        self.acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.prec = Precision(task='multiclass', num_classes=num_classes, average='macro')
        self.rec = Recall(task='multiclass', num_classes=num_classes, average='macro')
        self._per_class_f1 = MulticlassF1Score(num_classes=num_classes, average=None)
        self.class_weights = class_weights

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
        logits = self(x)
        loss = loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss_fn = nn.CrossEntropyLoss()
        logits = self(x)
        loss = loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.acc(preds, y))
        self.log("val_f1", self.f1(preds, y))
        self.log("val_precision", self.prec(preds, y))
        self.log("val_recall", self.rec(preds, y))
        return loss

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = {
            "Adam": optim.Adam,
            "AdamW": optim.AdamW,
            "RMSprop": optim.RMSprop
        }[self.hparams.optimizer](params, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer

# optuna hyperparameter tuning
from optuna.integration import PyTorchLightningPruningCallback

def objective(trial, train_loader, val_loader, num_classes, fixed_class_weights, output_dir):
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    fc_layers = trial.suggest_int('fc_layers', 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.6)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    fc_dim = trial.suggest_categorical("fc_hidden_dim", [256, 512, 1024])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop"])

    os.makedirs(os.path.join(output_dir, "trial_logs"), exist_ok=True)

    trial_logger = CSVLogger(
        save_dir=os.path.join(output_dir, "trial_logs"),
        name=f"trial_{trial.number}"
    )

    trial_train_loader = DataLoader(
        train_loader.dataset, batch_size=32, shuffle=True,
        num_workers=2, persistent_workers=True, worker_init_fn=worker_init_fn
    )
    trial_val_loader = DataLoader(
        val_loader.dataset, batch_size=32, shuffle=False,
        num_workers=2, persistent_workers=True, worker_init_fn=worker_init_fn
    )

    model = DenseNet121Lightning(
        num_classes, lr, dropout, weight_decay, fc_dim,
        optimizer_name, fixed_class_weights, fc_layers
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1", save_top_k=1, mode="max",
        filename="best-{epoch:02d}-valf1{val_f1:.4f}",
        dirpath=os.path.join(output_dir, f"trial_{trial.number}")
    )

    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_f1")

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator='gpu',
        devices=1,
        deterministic=True,
        logger=trial_logger,
        enable_progress_bar=True,
        precision="16-mixed",
        callbacks=[EarlyStopping(monitor="val_loss", patience=3), checkpoint_callback, pruning_callback]
    )

    trainer.fit(model, trial_train_loader, trial_val_loader)

    val_f1 = trainer.callback_metrics.get("val_f1")
    return val_f1.item() if val_f1 else float("-inf")

# main function 
import time 
def main():
    set_seed(42)

    dataset_root = r"C:/Users/Xuxu/Desktop/CCMT Dataset"
    output_dir = r"C:/Users/Xuxu/Desktop/Master Thesis/OptunaDensenetFull"

    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    train_ds, val_ds, all_labels, num_classes, class_to_idx, train_idx, val_idx, test_idx = setup_ssl_dataset(
        dataset_root=dataset_root,
        split_save_path=output_dir
    )

    np.save("C:/Users/Xuxu/Desktop/Master Thesis/train_indices.npy", train_idx)
    np.save("C:/Users/Xuxu/Desktop/Master Thesis/val_indices.npy", val_idx)
    np.save("C:/Users/Xuxu/Desktop/Master Thesis/test_indices.npy", test_idx)


# santiy checks for data leakage
    assert len(set(train_idx).intersection(val_idx)) == 0, "Train and validation splits overlap!"
    assert len(set(train_idx).intersection(test_idx)) == 0, "Train and test splits overlap!"
    assert len(set(val_idx).intersection(test_idx)) == 0, "Validation and test splits overlap!"
    print(" No data leakage between train, val, and test splits.")


# class distribution logging
    train_labels_for_weights = [all_labels[i] for i in train_idx]
    unique, counts = np.unique(train_labels_for_weights, return_counts=True)
    class_distribution = {int(cls): int(count) for cls, count in zip(unique, counts)}
    with open(os.path.join(output_dir, "class_distribution.json"), "w") as f:
        json.dump(class_distribution, f, indent=4)
    print("[INFO] Class distribution saved.")
    
    # save/load class weights for reproducibility
    weights_path = os.path.join(output_dir, "class_weights.pt")
    if os.path.exists(weights_path):
        class_weights = torch.load(weights_path)
        print("[INFO] Loaded class weights from file.")
    else:
        class_weights = torch.tensor(
            compute_class_weight('balanced', classes=np.unique(train_labels_for_weights), y=train_labels_for_weights),
            dtype=torch.float
        )
        torch.save(class_weights, weights_path)
        print("[INFO] Computed and saved class weights.")
    
    # create data loaders
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, persistent_workers=True, pin_memory=True , worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, persistent_workers=True, pin_memory=True , worker_init_fn=worker_init_fn)

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5, interval_steps=1)
    )

    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, num_classes, class_weights, output_dir),
        n_trials=15, show_progress_bar=True
    ) 

    print("Number of finished trials: ", len(study.trials))
    print("Number of pruned trials: ", len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]))
    print("Number of complete trials: ", len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]))

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")


    # save the study and best trial parameters
    joblib.dump(study, os.path.join(output_dir, "new_densenet_study.pkl"))
    with open(os.path.join(output_dir, "best_trial_params.json"), "w") as f:
        json.dump(study.best_trial.params, f, indent=4)

    elapsed_time = time.time() - start_time
    print(f" Total tuning time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
    print("Best hyperparameters:", study.best_trial.params)

if __name__ == "__main__":
    main()
