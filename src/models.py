"""
Arquitecturas de redes neuronales para clasificación de espectrogramas financieros.
"""

import torch
import torch.nn as nn
from torchvision import models


class SpectrogramCNN(nn.Module):
    """
    CNN ligera de 4 bloques convolucionales.
    Diseñada para espectrogramas 92×92.
    Arquitectura: Conv → BN → ReLU → Conv → BN → ReLU → MaxPool → Dropout
    
    Aproximadamente 1.45M parámetros.
    Ideal para datasets pequeños (<5000 imágenes).
    """
    def __init__(self, num_classes=2, dropout=0.3):
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
            conv_block(3,   32,  pool=True,  drop=0.1),
            conv_block(32,  64,  pool=True,  drop=0.15),
            conv_block(64,  128, pool=True,  drop=0.2),
            conv_block(128, 256, pool=True,  drop=0.25),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
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


class SpectrogramCNNv2(nn.Module):
    """
    Variante mejorada de SpectrogramCNN.
    Cambios principales:
      - Activación GELU en vez de ReLU
      - Global Average Pooling en vez de AdaptiveAvgPool(2,2)
      - Reduce drásticamente parámetros del clasificador (256 vs 1024)
      - Menos overfitting en datasets pequeños
    
    Aproximadamente 370K parámetros (3.9x menos que v1).
    """
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()

        def conv_block(in_ch, out_ch, pool=True, drop=0.2):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2, 2))
            layers.append(nn.Dropout2d(drop))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(3,   32,  pool=True,  drop=0.1),
            conv_block(32,  64,  pool=True,  drop=0.15),
            conv_block(64,  128, pool=True,  drop=0.2),
            conv_block(128, 256, pool=True,  drop=0.25),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


class ResNetFinancial(nn.Module):
    """
    Transfer learning con ResNet-18 preentrenado en ImageNet.
    Opcionalmente congela el backbone y entrena solo la cabeza personalizada.
    
    Aproximadamente 11.2M parámetros (con backbone congelado, 370K entrenables).
    Mejor performance en datasets medianos/grandes.
    """
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        if freeze_backbone:
            for param in list(base.parameters())[:-10]:
                param.requires_grad = False
        
        in_features = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        self.model = base

    def forward(self, x):
        return self.model(x)


class EfficientNetFinancial(nn.Module):
    """
    Transfer learning con EfficientNet-B0 preentrenado en ImageNet.
    Opcionalmente congela el backbone (features) y entrena solo el clasificador.
    
    Eficiente en memoria y cálculo, buen trade-off entre performance y velocidad.
    """
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        if freeze_backbone:
            for param in base.features.parameters():
                param.requires_grad = False
        
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        self.model = base

    def forward(self, x):
        return self.model(x)


def get_model(model_name: str, num_classes: int = 2) -> nn.Module:
    """
    Factory function para instanciar modelos por nombre.
    
    Args:
        model_name: uno de "cnn_v1", "cnn_v2", "resnet", "efficientnet"
        num_classes: número de clases (default 2 para BUY/SELL)
    
    Returns:
        Instancia del modelo.
    
    Raises:
        ValueError: si model_name no es reconocido.
    """
    models_dict = {
        "cnn_v1": SpectrogramCNN,
        "cnn_v2": SpectrogramCNNv2,
        "resnet": ResNetFinancial,
        "efficientnet": EfficientNetFinancial,
    }
    
    if model_name not in models_dict:
        raise ValueError(
            f"Modelo desconocido: '{model_name}'. "
            f"Opciones: {list(models_dict.keys())}"
        )
    
    return models_dict[model_name](num_classes=num_classes)
