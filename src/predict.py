#!/usr/bin/env python3
"""
Script de predicción en tiempo real para cualquier ticker financiero.

Descarga datos, genera espectrograma (STFT o Wavelet), carga modelo entrenado
e infiere la señal de compra/venta.

Uso:
  python predict.py --ticker BTC-USD --model resnet --weights ./models/resnet_best.pth
  python predict.py --ticker AAPL --model cnn_v1 --weights ./models/cnn_v1_best.pth --horizon 7
"""

import os
import sys
import argparse
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import torch
from PIL import Image as PILImage
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.ndimage import pad as np_pad

from models import get_model

warnings.filterwarnings("ignore")

# Importar funciones de preprocesamiento desde generate_dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_dataset import log_returns, zscore, make_stft_image, make_wavelet_image


class PredictionPipeline:
    """Pipeline de predicción end-to-end."""
    
    def __init__(self, model_name: str, weights_path: str,
                 dataset_type: str = "stft", num_classes: int = 2):
        self.model_name = model_name
        self.dataset_type = dataset_type
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = get_model(model_name, num_classes=num_classes)
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((92, 92)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        print(f"✅ Modelo cargado: {model_name}")
        print(f"   Pesos: {weights_path}")
        print(f"   Dispositivo: {self.device}")
    
    def download_ticker_data(self, ticker: str, days: int = 120) -> pd.DataFrame:
        """Descarga datos históricos de ticker."""
        print(f"\n📥 Descargando datos de {ticker} (últimos {days} días)...")
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            raise ValueError(f"No se encontraron datos para {ticker}")
        
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["Date"]).dt.date
        df = df.sort_values("date").reset_index(drop=True)
        
        print(f"   Datos descargados: {len(df)} días")
        print(f"   Rango: {df['date'].min()} a {df['date'].max()}")
        
        return df
    
    def preprocess_and_generate_image(self, df: pd.DataFrame, window_days: int = 90) -> np.ndarray:
        """Preprocesa datos y genera imagen espectrográmica."""
        close = df["Close"].values.astype(float)
        
        returns = log_returns(close)
        if np.any(np.isnan(returns)) or np.any(np.isinf(returns)):
            print("⚠️ Advertencia: NaN/Inf en retornos, usando ffill")
            returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        z_scores = zscore(returns)
        z_scores = np.nan_to_num(z_scores, nan=0.0, posinf=0.0, neginf=0.0)
        
        window = min(window_days, len(z_scores))
        z_window = z_scores[-window:]
        
        if len(z_window) < 20:
            raise ValueError(f"Ventana muy pequeña ({len(z_window)} días), necesita >20")
        
        print(f"   Preprocesamiento: {len(z_window)} valores normalizados")
        
        if self.dataset_type.lower() == "stft":
            img_array = make_stft_image(z_window)
            print(f"   Imagen STFT generada: {img_array.shape}")
        elif self.dataset_type.lower() == "wavelet":
            img_array = make_wavelet_image(z_window)
            print(f"   Imagen Wavelet generada: {img_array.shape}")
        else:
            raise ValueError(f"dataset_type inválido: {self.dataset_type}")
        
        return img_array
    
    def infer(self, img_array: np.ndarray):
        """Realiza inferencia."""
        if isinstance(img_array, np.ndarray):
            img = PILImage.fromarray((img_array * 255).astype(np.uint8), mode="RGB")
        else:
            img = img_array
        
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_idx = output.argmax(dim=1).item()
        
        return pred_idx, probs.cpu().numpy()
    
    def predict(self, ticker: str, horizon: int = 15, window_days: int = 90):
        """Pipeline completo de predicción."""
        try:
            df = self.download_ticker_data(ticker, days=120)
            img = self.preprocess_and_generate_image(df, window_days=window_days)
            pred_idx, probs = self.infer(img)
            
            classes = ["BUY", "SELL"]
            pred_class = classes[pred_idx]
            pred_prob = probs[pred_idx]
            
            print(f"\n{'='*60}")
            print(f"  PREDICCIÓN para {ticker} (horizonte {horizon}d)")
            print(f"{'='*60}")
            print(f"  Señal:        {pred_class}")
            print(f"  Confianza:    {pred_prob*100:.2f}%")
            print(f"  Modelo:       {self.model_name.upper()}")
            print(f"  Tipo:         {self.dataset_type.upper()}")
            print(f"\n  Probabilidades:")
            for cls, prob in zip(classes, probs):
                bar = "█" * int(prob * 30)
                print(f"    {cls:6s}: {prob*100:6.2f}% {bar}")
            print(f"{'='*60}\n")
            
            return {
                "ticker": ticker,
                "prediction": pred_class,
                "confidence": pred_prob,
                "probabilities": {cls: float(prob) for cls, prob in zip(classes, probs)},
                "model": self.model_name,
                "dataset_type": self.dataset_type,
                "horizon_days": horizon
            }
        
        except Exception as e:
            print(f"\n❌ Error durante predicción: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Predicción en tiempo real para cualquier ticker financiero",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python predict.py --ticker BTC-USD --model resnet --weights ./models/resnet_best.pth
  python predict.py --ticker AAPL --model cnn_v1 --weights ./models/cnn_v1_best.pth --horizon 7
  python predict.py --ticker GC=F --model efficientnet --weights ./models/efficientnet_best.pth \\
    --dataset-type wavelet --window 120
        """
    )
    
    parser.add_argument("--ticker", type=str, required=True,
                       help="Símbolo del ticker (ej: BTC-USD, AAPL, GC=F)")
    parser.add_argument("--model", type=str, default="resnet",
                       choices=["resnet", "efficientnet", "cnn_v1", "cnn_v2"],
                       help="Arquitectura del modelo (default: resnet)")
    parser.add_argument("--weights", type=str, required=True,
                       help="Ruta al archivo de pesos .pth")
    parser.add_argument("--horizon", type=int, default=15,
                       choices=[3, 7, 15, 30],
                       help="Horizonte de predicción en días (default: 15)")
    parser.add_argument("--dataset-type", type=str, default="stft",
                       choices=["stft", "wavelet"],
                       help="Tipo de espectrogram (default: stft)")
    parser.add_argument("--window", type=int, default=90,
                       help="Ventana temporal en días (default: 90)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.weights):
        print(f"❌ Archivo de pesos no encontrado: {args.weights}")
        sys.exit(1)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    pipeline = PredictionPipeline(
        model_name=args.model,
        weights_path=args.weights,
        dataset_type=args.dataset_type,
        num_classes=2
    )
    
    result = pipeline.predict(
        ticker=args.ticker,
        horizon=args.horizon,
        window_days=args.window
    )


if __name__ == "__main__":
    main()
