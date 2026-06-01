# =============================================================================
# CNN PARA CLASIFICACIÓN DE ESPECTROGRAMAS FINANCIEROS
# Continuación de financial_spectrogram.py
# Requiere haber ejecutado la Celda 9 del script anterior (dataset generado)
# =============================================================================

# ── CELDA 11: Instalación ─────────────────────────────────────────────────────
# !pip install torch torchvision scikit-learn matplotlib seaborn

# ── CELDA 12: Imports ─────────────────────────────────────────────────────────
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# Reproducibilidad
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Usando: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ── CELDA 13: Configuración global ───────────────────────────────────────────
DATASET_DIR  = "./spectrograms_dataset"
IMG_SIZE     = 64        # Debe coincidir con el tamaño generado antes
BATCH_SIZE   = 32
EPOCHS       = 40
LR           = 3e-4      # Learning rate inicial
LR_PATIENCE  = 5         # Epochs sin mejora para reducir LR
EARLY_STOP   = 12        # Epochs sin mejora para detener entrenamiento
CLASSES      = ["DOWN", "HOLD", "UP"]   # Orden alfabético = orden de PyTorch

# ── CELDA 14: Dataset custom, augmentation y loaders ─────────────────────────
#
# Cambios respecto al esquema anterior con ImageFolder:
#   1. Dataset custom que lee labels.csv  (no hay subcarpetas por clase)
#   2. Split por fecha, no aleatorio      (evita data leakage temporal)
#      train → fechas < VAL_CUTOFF
#      val   → fechas ≥ VAL_CUTOFF dentro del split "train" del CSV
#      test  → split "test" del CSV (todo 2026, nunca visto en entrenamiento)
#   3. HORIZON elige qué columna de etiqueta usar (1d, 5d, 10d, 20d, 30d)
#   4. CLASSES son 4: AVOID, BUY, HOLD, SELL  (orden alfabético fijo)
#
# ⚠️  NO flip horizontal → invierte el tiempo en el espectrograma
# ⚠️  NO flip vertical   → invierte el eje de frecuencias
# ⚠️  NO translate en Y  → desplaza frecuencias hacia arriba/abajo
# ✅  Translate solo en X (tiempo), ColorJitter: no rompen la física

import pandas as pd
from PIL import Image as PILImage
DATASET_DIR = "./dataset_stft"    # o "./dataset_wavelet"
HORIZON     = 10                  # 1, 5, 10, 20, 30
CLASSES     = ["AVOID", "BUY", "HOLD", "SELL"]
VAL_CUTOFF  = "2025-07-01"        # corte cronológico train / val

# ── Transforms ────────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomAffine(degrees=0, translate=(0.02, 0.0)),  # solo eje X
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ── Dataset custom ────────────────────────────────────────────────────────────
class SpectrogramDataset(Dataset):
    """
    Lee imágenes desde images/{split}/ y etiquetas desde labels.csv.
    El argumento horizon selecciona la columna label_{h}d del CSV.
    El mismo CSV sirve para entrenar modelos de distintos horizontes
    sin regenerar imágenes.
    """
    def __init__(self, df: pd.DataFrame, img_dir: str,
                 horizon: int, transform=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.label_col = f"label_{horizon}d"
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = PILImage.open(
            os.path.join(self.img_dir, row["filename"])
        ).convert("RGB")
        label = CLASSES.index(row[self.label_col])
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_labels(self) -> list:
        """Devuelve todas las etiquetas como enteros (para WeightedRandomSampler)."""
        return [CLASSES.index(lbl) for lbl in self.df[self.label_col]]


# ── Cargar CSV y separar splits por fecha ────────────────────────────────────
csv_path = os.path.join(DATASET_DIR, "labels.csv")
df_all   = pd.read_csv(csv_path, parse_dates=["date"])

df_train_full = df_all[df_all["split"] == "train"].copy()
df_test       = df_all[df_all["split"] == "test"].copy()

# Split cronológico: val = últimos meses del período train
df_train = df_train_full[df_train_full["date"] <  VAL_CUTOFF]
df_val   = df_train_full[df_train_full["date"] >= VAL_CUTOFF]

img_train_dir = os.path.join(DATASET_DIR, "images", "train")
img_test_dir  = os.path.join(DATASET_DIR, "images", "test")

train_ds = SpectrogramDataset(df_train, img_train_dir, HORIZON, train_tf)
val_ds   = SpectrogramDataset(df_val,   img_train_dir, HORIZON, eval_tf)
test_ds  = SpectrogramDataset(df_test,  img_test_dir,  HORIZON, eval_tf)

# ── WeightedRandomSampler (desbalance de clases) ──────────────────────────────
train_labels  = train_ds.get_labels()
class_counts  = Counter(train_labels)
total_train   = len(train_labels)
class_weights = {cls: total_train / max(count, 1)
                 for cls, count in class_counts.items()}
sample_weights = [class_weights[lbl] for lbl in train_labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

# ── DataLoaders ───────────────────────────────────────────────────────────────
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)

# ── Resumen ───────────────────────────────────────────────────────────────────
print(f"📦 Dataset    : {DATASET_DIR}")
print(f"   Horizonte  : {HORIZON}d  →  columna label_{HORIZON}d")
print(f"   Train      : {len(train_ds):6d}  (< {VAL_CUTOFF})")
print(f"   Val        : {len(val_ds):6d}  (≥ {VAL_CUTOFF}, dentro de train)")
print(f"   Test       : {len(test_ds):6d}  (2026, nunca visto en entrenamiento)")
print(f"\n📊 Distribución en train (horizonte {HORIZON}d):")
for idx, cls in enumerate(CLASSES):
    n   = class_counts.get(idx, 0)
    pct = n / total_train * 100
    bar = "█" * int(pct / 2)
    print(f"   {cls:5s}: {n:5d} ({pct:4.1f}%)  {bar}")

# ── CELDA 15: Arquitectura CNN ────────────────────────────────────────────────
# Dos opciones. Elegí una descomentando el bloque correspondiente.

# ─── OPCIÓN A: CNN Custom liviana (ideal si el dataset es pequeño < 5000 imgs)

class SpectrogramCNN(nn.Module):
    """
    CNN de 4 bloques convolucionales diseñada para espectrogramas 64×64.
    Cada bloque: Conv → BN → ReLU → Conv → BN → ReLU → MaxPool → Dropout
    """
    def __init__(self, num_classes=3, dropout=0.4):
        super().__init__()

        def conv_block(in_ch, out_ch, pool=True, drop=0.2):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2, 2))
            layers.append(nn.Dropout2d(drop))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(3,   32,  pool=True,  drop=0.1),   # → 32×32
            conv_block(32,  64,  pool=True,  drop=0.15),  # → 16×16
            conv_block(64,  128, pool=True,  drop=0.2),   # → 8×8
            conv_block(128, 256, pool=True,  drop=0.25),  # → 4×4
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),    # → 256×2×2 = 1024
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─── OPCIÓN B: Transfer Learning con ResNet18 (mejor si el dataset es grande)
# Descomenta este bloque y comenta la instanciación de SpectrogramCNN abajo

# class ResNetFinancial(nn.Module):
#     def __init__(self, num_classes=3, freeze_backbone=True):
#         super().__init__()
#         base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#         if freeze_backbone:
#             for param in list(base.parameters())[:-10]:
#                 param.requires_grad = False
#         in_features = base.fc.in_features
#         base.fc = nn.Sequential(
#             nn.Linear(in_features, 128),
#             nn.ReLU(),
#             nn.Dropout(0.4),
#             nn.Linear(128, num_classes)
#         )
#         self.model = base
#     def forward(self, x):
#         return self.model(x)

# ── Instanciar modelo ────────────────────────────────────────────────────────
model = SpectrogramCNN(num_classes=len(CLASSES)).to(DEVICE)
# model = ResNetFinancial(num_classes=len(CLASSES)).to(DEVICE)   # ← Opción B

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(model)
print(f"\n🔢 Parámetros totales     : {total_params:,}")
print(f"   Parámetros entrenables : {trainable_params:,}")

# ── CELDA 16: Loss, optimizador, schedulers ───────────────────────────────────
# Label smoothing para evitar que la red sea demasiado confiada
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

# Reduce LR cuando val_loss no mejora
scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=LR_PATIENCE, verbose=True
)

# Warmup + cosine decay (opcional, descomenta si usás ResNet)
# scheduler_cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)

print("✅ Optimizador y scheduler configurados")
print(f"   LR inicial: {LR}  |  Weight decay: 1e-4  |  Label smoothing: 0.1")

# ── CELDA 17: Funciones de entrenamiento y evaluación ────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Evita exploding gradients
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds      = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        probs   = torch.softmax(outputs, dim=1)
        preds   = outputs.argmax(dim=1)
        total_loss += loss.item() * imgs.size(0)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels, all_probs

# ── CELDA 18: Loop de entrenamiento ──────────────────────────────────────────
history = {
    "train_loss": [], "val_loss": [],
    "train_acc":  [], "val_acc":  [],
    "lr":         []
}

best_val_loss = float("inf")
best_weights  = None
patience_ctr  = 0

print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'LR':>10}")
print("─" * 65)

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc           = train_epoch(model, train_loader, criterion, optimizer)
    vl_loss, vl_acc, _, _, _  = eval_epoch(model, val_loader, criterion)

    scheduler_plateau.step(vl_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(vl_acc)
    history["lr"].append(current_lr)

    # Checkpoint del mejor modelo
    if vl_loss < best_val_loss:
        best_val_loss = vl_loss
        best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
        patience_ctr  = 0
        flag = " ◀ best"
    else:
        patience_ctr += 1
        flag = f"  (no mejora {patience_ctr}/{EARLY_STOP})"

    print(f"{epoch:>6} | {tr_loss:>10.4f} | {tr_acc*100:>8.2f}% | {vl_loss:>8.4f} | {vl_acc*100:>6.2f}% | {current_lr:>10.2e}{flag}")

    if patience_ctr >= EARLY_STOP:
        print(f"\n⏹  Early stopping en epoch {epoch}")
        break

# Restaurar mejores pesos
model.load_state_dict(best_weights)
torch.save(best_weights, "spectrogram_cnn_best.pth")
print(f"\n✅ Mejor modelo guardado  |  Val Loss: {best_val_loss:.4f}")

# ── CELDA 19: Curvas de entrenamiento ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("#0f0f23")
fig.suptitle("Historial de entrenamiento", color="white", fontsize=14, fontweight="bold")

ep = range(1, len(history["train_loss"]) + 1)
colors = {"train": "#3498db", "val": "#e74c3c"}

# Loss
axes[0].plot(ep, history["train_loss"], color=colors["train"], label="Train", linewidth=1.8)
axes[0].plot(ep, history["val_loss"],   color=colors["val"],   label="Val",   linewidth=1.8)
axes[0].set_title("Loss", color="white")
axes[0].set_xlabel("Epoch", color="gray")
axes[0].legend(facecolor="#1a1a2e", labelcolor="white")

# Accuracy
axes[1].plot(ep, [a * 100 for a in history["train_acc"]], color=colors["train"], label="Train", linewidth=1.8)
axes[1].plot(ep, [a * 100 for a in history["val_acc"]],   color=colors["val"],   label="Val",   linewidth=1.8)
axes[1].set_title("Accuracy (%)", color="white")
axes[1].set_xlabel("Epoch", color="gray")
axes[1].legend(facecolor="#1a1a2e", labelcolor="white")

# Learning rate
axes[2].semilogy(ep, history["lr"], color="#2ecc71", linewidth=1.8)
axes[2].set_title("Learning Rate", color="white")
axes[2].set_xlabel("Epoch", color="gray")

for ax in axes:
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="gray")
    ax.spines[:].set_color("#333355")
    ax.grid(True, alpha=0.12, color="white")

plt.tight_layout()
plt.show()

# ── CELDA 20: Evaluación en Test Set ─────────────────────────────────────────
test_loss, test_acc, preds, labels, probs = eval_epoch(model, test_loader, criterion)
probs  = np.array(probs)
preds  = np.array(preds)
labels = np.array(labels)

print(f"\n{'='*50}")
print(f"  TEST LOSS     : {test_loss:.4f}")
print(f"  TEST ACCURACY : {test_acc*100:.2f}%")
print(f"{'='*50}\n")
print(classification_report(labels, preds, target_names=CLASSES, digits=3))

# ── CELDA 21: Matriz de confusión ─────────────────────────────────────────────
cm = confusion_matrix(labels, preds, normalize="true")

fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor("#0f0f23")
ax.set_facecolor("#1a1a2e")

sns.heatmap(
    cm, annot=True, fmt=".2f", cmap="Blues",
    xticklabels=CLASSES, yticklabels=CLASSES,
    ax=ax, linewidths=0.5, linecolor="#333355",
    annot_kws={"size": 14, "color": "white"}
)
ax.set_title("Matriz de Confusión (normalizada)", color="white", fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Predicción", color="gray", fontsize=11)
ax.set_ylabel("Real", color="gray", fontsize=11)
ax.tick_params(colors="gray")

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(colors="gray")

plt.tight_layout()
plt.show()

# ── CELDA 22: Curvas ROC por clase ────────────────────────────────────────────
labels_bin = label_binarize(labels, classes=[0, 1, 2])
palette    = ["#e74c3c", "#f39c12", "#2ecc71"]

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor("#0f0f23")
ax.set_facecolor("#1a1a2e")

for i, (cls, color) in enumerate(zip(CLASSES, palette)):
    fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{cls} (AUC = {roc_auc:.3f})")

ax.plot([0, 1], [0, 1], "w--", linewidth=0.8, alpha=0.4, label="Aleatorio")
ax.set_xlabel("False Positive Rate", color="gray")
ax.set_ylabel("True Positive Rate", color="gray")
ax.set_title("Curvas ROC por clase", color="white", fontsize=13, fontweight="bold")
ax.legend(facecolor="#1a1a2e", labelcolor="white")
ax.tick_params(colors="gray")
ax.spines[:].set_color("#333355")
ax.grid(True, alpha=0.12, color="white")

plt.tight_layout()
plt.show()

# ── CELDA 23: Grad-CAM — ¿Qué parte del espectrograma mira la red? ────────────
# Implementación manual de Grad-CAM para la última capa convolucional

class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor, class_idx):
        self.model.eval()
        output = self.model(img_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = torch.clamp(cam, min=0)
        cam     = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()


# Acceder a la última capa conv del modelo custom
target_layer = list(model.features[-1].children())[-2]  # Último Conv2d
grad_cam     = GradCAM(model, target_layer)

# Tomar 6 ejemplos del test set
sample_imgs, sample_labels = next(iter(DataLoader(test_ds, batch_size=6, shuffle=True)))
sample_imgs  = sample_imgs.to(DEVICE)
sample_preds = model(sample_imgs).argmax(dim=1).cpu()

fig, axes = plt.subplots(2, 6, figsize=(18, 7))
fig.patch.set_facecolor("#0f0f23")
fig.suptitle("Grad-CAM — Zonas del espectrograma que activan la decisión", color="white", fontsize=13, fontweight="bold")

unnorm = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])

for col in range(6):
    img_t  = sample_imgs[col:col+1]
    pred   = sample_preds[col].item()
    true   = sample_labels[col].item()
    cam    = grad_cam.generate(img_t, pred)

    # Imagen original
    img_show = unnorm(sample_imgs[col]).cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    axes[0, col].imshow(img_show)
    axes[0, col].axis("off")
    color = "#2ecc71" if pred == true else "#e74c3c"
    axes[0, col].set_title(f"Real: {CLASSES[true]}\nPred: {CLASSES[pred]}", color=color, fontsize=8)

    # Grad-CAM superpuesto
    import cv2
    cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heatmap     = plt.cm.jet(cam_resized)[:, :, :3]
    overlay     = 0.55 * img_show + 0.45 * heatmap
    axes[1, col].imshow(np.clip(overlay, 0, 1))
    axes[1, col].axis("off")
    axes[1, col].set_title("Grad-CAM", color="gray", fontsize=8)

for ax in axes.flat:
    ax.set_facecolor("#1a1a2e")

plt.tight_layout()
plt.show()

# ── CELDA 24: Predicción en tiempo real sobre un nuevo ticker ─────────────────
import yfinance as yf
from scipy import signal as sig

def predecir_ticker(ticker, window_days=60):
    """Descarga los últimos datos y predice la dirección del próximo movimiento."""
    df_new  = yf.download(ticker, period="6mo", interval="1d", progress=False)
    close   = df_new["Close"].squeeze().values
    returns = np.diff(np.log(close))
    ret_n   = (returns - returns.mean()) / (returns.std() + 1e-8)

    if len(ret_n) < window_days:
        print(f"⚠️  No hay suficientes datos para {ticker}")
        return

    window = ret_n[-window_days:]
    f_w, t_w, S_w = sig.spectrogram(
        window, fs=1.0,
        window=sig.windows.hann(min(32, window_days)),
        noverlap=min(28, window_days - 1),
        nfft=64, scaling="density"
    )
    img_data = 10 * np.log10(S_w + 1e-12)

    # Convertir a tensor
    fig_tmp, ax_tmp = plt.subplots(figsize=(1, 1), dpi=64)
    ax_tmp.pcolormesh(t_w, f_w, img_data, cmap="magma", shading="gouraud")
    ax_tmp.axis("off")
    fig_tmp.subplots_adjust(left=0, right=1, top=1, bottom=0)
    tmp_path = f"/tmp/{ticker}_latest.png"
    fig_tmp.savefig(tmp_path, dpi=64, bbox_inches="tight", pad_inches=0)
    plt.close(fig_tmp)

    from PIL import Image
    img   = Image.open(tmp_path).convert("RGB")
    img_t = val_tf(img).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(img_t)
        probs_ = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred   = logits.argmax(dim=1).item()

    print(f"\n{'='*45}")
    print(f"  🎯 Ticker        : {ticker}")
    print(f"  📅 Ventana       : últimos {window_days} días")
    print(f"  🔮 Predicción    : {CLASSES[pred]}")
    print(f"{'─'*45}")
    for cls, p in zip(CLASSES, probs_):
        bar = "█" * int(p * 30)
        print(f"  {cls:5s}: {bar:<30} {p*100:5.1f}%")
    print(f"{'='*45}")

    # Mostrar el espectrograma que se usó
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    fig2.patch.set_facecolor("#0f0f23")
    axes2[0].plot(df_new["Close"].squeeze().values[-window_days:], color="#2ecc71", linewidth=1.5)
    axes2[0].set_title(f"{ticker} — últimos {window_days} días", color="white")
    axes2[0].set_facecolor("#1a1a2e")
    axes2[0].tick_params(colors="gray")
    axes2[0].spines[:].set_color("#333355")

    axes2[1].pcolormesh(t_w, f_w, img_data, cmap="magma", shading="gouraud")
    axes2[1].set_title(f"Espectrograma  →  Predicción: {CLASSES[pred]}", color="white")
    axes2[1].set_facecolor("#1a1a2e")
    axes2[1].tick_params(colors="gray")
    axes2[1].spines[:].set_color("#333355")

    plt.tight_layout()
    plt.show()


# Probar con distintos activos
predecir_ticker("BTC-USD")
predecir_ticker("AAPL")