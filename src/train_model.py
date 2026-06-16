#!/usr/bin/env python3
"""
Entrenamiento y evaluación de modelos CNN para clasificación de espectrogramas financieros.

Soporta 4 arquitecturas diferentes:
  - cnn_v1:       CNN ligera personalizada (1.45M parámetros)
  - cnn_v2:       CNN mejorada con GELU + Global Average Pooling (370K parámetros)
  - resnet:       ResNet-18 transfer learning (11.2M parámetros)
  - efficientnet: EfficientNet-B0 transfer learning (eficiente)

Uso:
  python train_model.py --model resnet --dataset ./dataset_stft --horizon 15 --epochs 20
  python train_model.py --model all --epochs 20
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_curve, auc, label_binarize
)
from PIL import Image as PILImage

from models import get_model

warnings.filterwarnings("ignore")


class SpectrogramDataset(Dataset):
    """Dataset personalizado para espectrogramas financieros."""
    def __init__(self, df: pd.DataFrame, img_dir: str,
                 horizon: int, transform=None, classes=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.label_col = f"label_{horizon}d"
        self.transform = transform
        self.classes = classes or ["BUY", "SELL"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = PILImage.open(
            os.path.join(self.img_dir, row["filename"])
        ).convert("RGB")
        label = self.classes.index(row[self.label_col])
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_labels(self):
        return [self.classes.index(lbl) for lbl in self.df[self.label_col]]


def setup_device():
    """Detecta GPU disponible."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Dispositivo: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    return device


def create_transforms(img_size):
    """Crea transformaciones de imagen."""
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomAffine(degrees=0, translate=(0.02, 0.0)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    return train_tf, eval_tf


def load_datasets(dataset_dir, horizon, img_size, val_cutoff, classes):
    """Carga datasets con split temporal."""
    csv_path = os.path.join(dataset_dir, "labels.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset no encontrado: {csv_path}")
    
    df_all = pd.read_csv(csv_path, parse_dates=["date"])
    
    df_train_full = df_all[df_all["split"] == "train"].copy()
    df_test = df_all[df_all["split"] == "test"].copy()
    
    df_train = df_train_full[df_train_full["date"] < val_cutoff]
    df_val = df_train_full[df_train_full["date"] >= val_cutoff]
    
    label_col = f"label_{horizon}d"
    df_train = df_train[df_train[label_col] != "HOLD"].copy()
    df_val = df_val[df_val[label_col] != "HOLD"].copy()
    df_test = df_test[df_test[label_col] != "HOLD"].copy()
    
    train_tf, eval_tf = create_transforms(img_size)
    
    img_train_dir = os.path.join(dataset_dir, "images", "train")
    img_test_dir = os.path.join(dataset_dir, "images", "test")
    
    train_ds = SpectrogramDataset(df_train, img_train_dir, horizon, train_tf, classes)
    val_ds = SpectrogramDataset(df_val, img_train_dir, horizon, eval_tf, classes)
    test_ds = SpectrogramDataset(df_test, img_test_dir, horizon, eval_tf, classes)
    
    return train_ds, val_ds, test_ds


def create_loaders(train_ds, val_ds, test_ds, batch_size):
    """Crea DataLoaders con WeightedRandomSampler."""
    train_labels = train_ds.get_labels()
    class_counts = Counter(train_labels)
    total_train = len(train_labels)
    class_weights = {
        cls: total_train / max(count, 1)
        for cls, count in class_counts.items()
    }
    sample_weights = [class_weights[lbl] for lbl in train_labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_counts


def train_epoch(model, loader, criterion, optimizer, device):
    """Entrena un epoch."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    """Evalúa un epoch."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * imgs.size(0)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels, all_probs


def train_model(model_name, args, device):
    """Entrena un modelo completo."""
    print(f"\n{'='*70}")
    print(f"  Entrenando modelo: {model_name.upper()}")
    print(f"{'='*70}")
    
    try:
        train_ds, val_ds, test_ds = load_datasets(
            args.dataset, args.horizon, args.img_size,
            args.val_cutoff, args.classes
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None
    
    train_loader, val_loader, test_loader, class_counts = create_loaders(
        train_ds, val_ds, test_ds, args.batch_size
    )
    
    print(f"📦 Dataset: {args.dataset}")
    print(f"   Train: {len(train_ds):6d} | Val: {len(val_ds):6d} | Test: {len(test_ds):6d}")
    print(f"   Clases: {args.classes}")
    print(f"\n📊 Distribución (train):")
    for cls_idx, cls in enumerate(args.classes):
        count = class_counts.get(cls_idx, 0)
        pct = count / len(train_ds) * 100 if len(train_ds) > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"   {cls:10s}: {count:6d} ({pct:5.1f}%) {bar}")
    
    model = get_model(model_name, num_classes=len(args.classes)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🔢 Parámetros: {total_params:,} (entrenables: {trainable_params:,})")
    
    total = sum(class_counts.values())
    weights = torch.tensor(
        [total / max(class_counts.get(i, 1), 1) for i in range(len(args.classes))],
        dtype=torch.float32
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience
    )
    
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "val_f1": [], "lr": []
    }
    
    best_f1 = -1.0
    best_weights = None
    patience_ctr = 0
    
    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | "
          f"{'Val Acc':>8} | {'F1 Macro':>8} | {'LR':>10}")
    print("─" * 75)
    
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, preds, labels, _ = eval_epoch(model, val_loader, criterion, device)
        
        val_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        scheduler.step(vl_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
            flag = f" ◀ best F1={val_f1:.4f}"
        else:
            patience_ctr += 1
            flag = f"  (no mejora {patience_ctr}/{args.early_stop})"
        
        print(
            f"{epoch:>6} | {tr_loss:>10.4f} | {vl_loss:>10.4f} | "
            f"{vl_acc*100:>7.2f}% | {val_f1:>8.4f} | {current_lr:>10.2e}{flag}"
        )
        
        if patience_ctr >= args.early_stop:
            print(f"\n⏹ Early stopping en epoch {epoch}")
            break
    
    if best_weights:
        model.load_state_dict(best_weights)
    
    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, f"{model_name}_best.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ Modelo guardado: {model_path}")
    
    if not args.no_plots:
        plot_training_history(history, model_name, args.output)
    
    test_loss, test_acc, preds, labels, probs = eval_epoch(
        model, test_loader, criterion, device
    )
    
    print(f"\n{'='*50}")
    print(f"  TEST LOSS     : {test_loss:.4f}")
    print(f"  TEST ACCURACY : {test_acc*100:.2f}%")
    print(f"{'='*50}")
    print(classification_report(labels, preds, target_names=args.classes, digits=3))
    
    if not args.no_plots:
        plot_confusion_matrix(labels, preds, args.classes, model_name, args.output)
        if len(args.classes) == 2:
            plot_roc_curve(labels, probs, args.classes, model_name, args.output)
    
    return {
        "model_name": model_name,
        "best_f1": best_f1,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "model_params": total_params,
        "model_path": model_path
    }


def plot_training_history(history, model_name, output_dir):
    """Guarda gráficas de entrenamiento."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0f0f23")
    fig.suptitle(f"Historial: {model_name.upper()}", color="white", fontsize=14, fontweight="bold")
    
    ep = range(1, len(history["train_loss"]) + 1)
    colors = {"train": "#3498db", "val": "#e74c3c"}
    
    axes[0].plot(ep, history["train_loss"], color=colors["train"], label="Train", linewidth=1.8)
    axes[0].plot(ep, history["val_loss"], color=colors["val"], label="Val", linewidth=1.8)
    axes[0].set_title("Loss", color="white")
    axes[0].set_xlabel("Epoch", color="gray")
    axes[0].legend(facecolor="#1a1a2e", labelcolor="white")
    
    axes[1].plot(ep, [a * 100 for a in history["train_acc"]], color=colors["train"], label="Train", linewidth=1.8)
    axes[1].plot(ep, [a * 100 for a in history["val_acc"]], color=colors["val"], label="Val", linewidth=1.8)
    axes[1].set_title("Accuracy (%)", color="white")
    axes[1].set_xlabel("Epoch", color="gray")
    axes[1].legend(facecolor="#1a1a2e", labelcolor="white")
    
    axes[2].plot(ep, history["val_f1"], color="#2ecc71", linewidth=1.8)
    axes[2].set_title("F1 Macro", color="white")
    axes[2].set_xlabel("Epoch", color="gray")
    
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="gray")
        ax.spines[:].set_color("#333355")
        ax.grid(True, alpha=0.12, color="white")
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{model_name}_training.png")
    plt.savefig(plot_path, dpi=100, facecolor="#0f0f23")
    plt.close()
    print(f"   Gráfica guardada: {plot_path}")


def plot_confusion_matrix(labels, preds, classes, model_name, output_dir):
    """Guarda matriz de confusión."""
    cm = confusion_matrix(labels, preds, normalize="true")
    
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0f0f23")
    ax.set_facecolor("#1a1a2e")
    
    sns.heatmap(
        cm, annot=True, fmt=".2f",
        xticklabels=classes, yticklabels=classes,
        ax=ax, linewidths=0.5, linecolor="#333355",
        annot_kws={"size": 12, "color": "white"}
    )
    ax.set_title(f"Matriz de Confusión ({model_name.upper()})", color="white", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicción", color="gray")
    ax.set_ylabel("Real", color="gray")
    ax.tick_params(colors="gray")
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{model_name}_confusion.png")
    plt.savefig(plot_path, dpi=100, facecolor="#0f0f23")
    plt.close()
    print(f"   Matriz guardada: {plot_path}")


def plot_roc_curve(labels, probs, classes, model_name, output_dir):
    """Guarda curva ROC."""
    labels_bin = label_binarize(labels, classes=[0, 1])
    palette = ["#e74c3c", "#2ecc71"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0f0f23")
    ax.set_facecolor("#1a1a2e")
    
    for i, (cls, color) in enumerate(zip(classes, palette)):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{cls} (AUC = {roc_auc:.3f})")
    
    ax.plot([0, 1], [0, 1], "w--", linewidth=0.8, alpha=0.4, label="Aleatorio")
    ax.set_xlabel("False Positive Rate", color="gray")
    ax.set_ylabel("True Positive Rate", color="gray")
    ax.set_title(f"Curvas ROC ({model_name.upper()})", color="white", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    ax.tick_params(colors="gray")
    ax.spines[:].set_color("#333355")
    ax.grid(True, alpha=0.12, color="white")
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{model_name}_roc.png")
    plt.savefig(plot_path, dpi=100, facecolor="#0f0f23")
    plt.close()
    print(f"   ROC guardada: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Entrena modelos CNN para clasificación de espectrogramas financieros",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python train_model.py --model resnet --epochs 20
  python train_model.py --model all --epochs 20
  python train_model.py --model cnn_v1 --dataset ./dataset_wavelet --horizon 15
        """
    )
    
    parser.add_argument("--model", type=str, default="resnet",
                       choices=["resnet", "efficientnet", "cnn_v1", "cnn_v2", "all"],
                       help="Arquitectura a entrenar (default: resnet)")
    parser.add_argument("--dataset", type=str, default="./dataset_stft",
                       help="Directorio del dataset (default: ./dataset_stft)")
    parser.add_argument("--horizon", type=int, default=15,
                       choices=[3, 7, 15, 30],
                       help="Horizonte de predicción en días (default: 15)")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Número de epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Batch size (default: 256)")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate inicial (default: 3e-4)")
    parser.add_argument("--lr-patience", type=int, default=7,
                       help="Patience para ReduceLROnPlateau (default: 7)")
    parser.add_argument("--early-stop", type=int, default=10,
                       help="Patience para early stopping (default: 10)")
    parser.add_argument("--img-size", type=int, default=92,
                       help="Tamaño de imagen (default: 92)")
    parser.add_argument("--output", type=str, default="./models",
                       help="Directorio de salida para modelos (default: ./models)")
    parser.add_argument("--classes", nargs="+", default=["BUY", "SELL"],
                       help="Clases a predecir (default: BUY SELL)")
    parser.add_argument("--val-cutoff", type=str, default="2025-02-01",
                       help="Fecha de corte train/val (default: 2025-02-01)")
    parser.add_argument("--no-plots", action="store_true",
                       help="No guardar gráficas")
    
    args = parser.parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = setup_device()
    
    models_to_train = (
        ["resnet", "efficientnet", "cnn_v1", "cnn_v2"]
        if args.model == "all"
        else [args.model]
    )
    
    results = []
    for model_name in models_to_train:
        result = train_model(model_name, args, device)
        if result:
            results.append(result)
    
    if results and len(results) > 1:
        print(f"\n{'='*70}")
        print("  RESUMEN DE RESULTADOS")
        print(f"{'='*70}")
        print(f"{'Modelo':<15} | {'Best F1':<10} | {'Test Acc':<10} | {'Parámetros':<15}")
        print("─" * 70)
        for r in sorted(results, key=lambda x: x["best_f1"], reverse=True):
            print(
                f"{r['model_name']:<15} | {r['best_f1']:<10.4f} | "
                f"{r['test_acc']*100:<9.2f}% | {r['model_params']:>14,}"
            )


if __name__ == "__main__":
    main()
