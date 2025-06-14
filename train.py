import torch
import torch.nn as nn
import yaml
import pandas as pd
import os
from tqdm import tqdm
from data.cifar10_noise_loader import get_dataloaders
from models.custom_resnet import CustomResNet
from losses.focal_loss import FocalLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import (
    set_seed, initialize_weights, evaluate_model,
    save_checkpoint, classification_metrics
)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

class Trainer:
    def __init__(self, config):
        set_seed(config["seed"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader, self.val_loader = get_dataloaders(
            batch_size=config["batch_size"],
            noise_ratio=config["noise_ratio"],
            augment=True
        )
        self.model = CustomResNet().to(self.device)
        initialize_weights(self.model)

        self.criterion = FocalLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=3, factor=0.5, verbose=False)
        self.clip_norm = config["clip_grad_norm"]
        self.epochs = config["epochs"]
        self.save_path = config["save_path"]
        self.early_stopping = EarlyStopping(patience=5)
        self.history = []

    def train(self):
        best_acc = 0.0
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for images, labels in progress:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                self.optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.train_loader)
            val_loss, val_acc, val_f1, val_precision, val_recall = classification_metrics(
                self.model, self.val_loader, self.device, self.criterion
            )
            self.scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.4f} - F1: {val_f1:.4f}")

            self.history.append({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_precision": val_precision,
                "val_recall": val_recall
            })

            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(self.model, self.save_path)

            if self.early_stopping.step(val_loss):
                print(f"Early stopping at epoch {epoch+1}")
                break

        df_hist = pd.DataFrame(self.history)
        df_hist.to_csv("training_metrics.csv", index=False)
        print("training_metrics.csv saved with per-epoch metrics")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)
    trainer = Trainer(config)
    trainer.train()
