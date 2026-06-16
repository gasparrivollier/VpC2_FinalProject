#!/usr/bin/env python3
"""
=============================================================================
 GENERADOR DE DATASETS DE ESPECTROGRAMAS FINANCIEROS
 Multi-activo: crypto, commodities, stocks, FX
 Horizontes de etiquetado: 1d, 5d, 10d, 20d, 30d
 Etiquetas: BUY / HOLD / SELL  (umbrales adaptativos por ATR)
 2026 reservado como test set

 Genera DOS datasets con la MISMA estructura y etiquetas:
   dataset_stft/    — imágenes basadas en Short-Time Fourier Transform
   dataset_wavelet/ — imágenes basadas en Continuous Wavelet Transform (Morlet)

 Estructura de cada dataset:
   images/
     train/   BTC-USD_00060.png ...
     test/    ...
   labels.csv
     sample_id, filename, ticker, asset_class, date, split, atr,
     ret_1d,  ret_5d,  ret_10d,  ret_20d,  ret_30d,
     label_1d, label_5d, label_10d, label_20d, label_30d
=============================================================================
"""

import os
import csv
import time
import json
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import pywt
import yfinance as yf
from scipy import signal
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm

warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════════════════════════════════════
# UNIVERSO DE ACTIVOS
# ═════════════════════════════════════════════════════════════════════════════

ASSET_UNIVERSE = {
    "crypto": [
        "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD"
    ],
    "stocks": [
        "AAPL", "MSFT", "GOOGL", "SPY", "NVDA", "KO", "TSLA", "AMD", "META", "NFLX"
    ],
    "commodities": [
        "GC=F", "SI=F", "CL=F"
    ],
    "fx": [
        "EURUSD=X",
    ],
}

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═════════════════════════════════════════════════════════════════════════════

HORIZONS    = [3, 7, 15, 30]
WINDOW_DAYS = 90        # días de historia para cada muestra
IMG_SIZE    = 92        # píxeles de salida
CMAP        = "magma"
PAD_LENGTH  = 16        # días de padding por reflexión
START_DATE  = "2021-01-01"
TRAIN_END   = "2025-12-31"
TEST_START  = "2026-01-01"
MIN_HISTORY = 200       # mínimo de días para procesar un activo

# Configuración STFT
STFT_CFG = {
    "stft_window":  32,   # tamaño de ventana Hann
    "stft_overlap": 28,   # solapamiento entre ventanas
    "nfft":         64,   # resolución FFT (zero-padding)
}

# Configuración Wavelet de Morlet continua
# cmor{B}-{C}: B = bandwidth (ancho de banda), C = frecuencia central
# cmor1.5-1.0 es el estándar en análisis financiero
# scales: cada valor s representa un ciclo de aproximadamente s días
WAVELET_CFG = {
    "wavelet": "cmor1.5-1.0",
    "scales":  list(range(2, 31)),  # ciclos de 2 a 30 días
}

ATR_CFG = {
    "period":           14,
    "buy_multiplier":  0.7,
    "sell_multiplier": 0.7,
}


# ═════════════════════════════════════════════════════════════════════════════
# PREPROCESAMIENTO DE SEÑAL
# ═════════════════════════════════════════════════════════════════════════════

def log_returns(close: np.ndarray) -> np.ndarray:
    """r_t = log(P_t / P_{t-1})  —  serie estacionaria y aditiva."""
    return np.diff(np.log(close))


def zscore(x: np.ndarray) -> np.ndarray:
    """Normalización Z. Media 0, std 1. Hace comparables activos distintos."""
    std = x.std()
    return (x - x.mean()) / std if std > 1e-10 else np.zeros_like(x)


def compute_atr(high: np.ndarray, low: np.ndarray,
                close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    ATR como porcentaje del precio (EMA).
    Umbral dinámico: refleja la volatilidad real del activo en ese momento.
    Lo escalamos por sqrt(horizon) al etiquetar porque la volatilidad
    crece con la raíz del tiempo.
    """
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        )
    )
    atr_pct = tr / close[:-1]
    atr = np.full_like(atr_pct, np.nan)
    atr[period - 1] = atr_pct[:period].mean()
    alpha = 2.0 / (period + 1)
    for i in range(period, len(atr_pct)):
        atr[i] = alpha * atr_pct[i] + (1 - alpha) * atr[i - 1]
    return atr


def make_label(future_return: float, threshold: float) -> str:
    """
    BUY  →  retorno > +threshold   (señal alcista fuerte)
    SELL →  retorno < -threshold   (señal bajista fuerte)
    HOLD →  cualquier cosa en el medio
    """
    if   future_return >  threshold: return "BUY"
    elif future_return < -threshold: return "SELL"
    else:                            return "HOLD"


# ═════════════════════════════════════════════════════════════════════════════
# GENERACIÓN DE IMÁGENES
# ═════════════════════════════════════════════════════════════════════════════

def make_stft_image(window: np.ndarray) -> np.ndarray:
    """
    STFT: aplica FFT sobre ventanas deslizantes con forma de Hann.
    Resolución tiempo-frecuencia fija para todas las frecuencias.
    Retorna matriz 2D en dB: (n_frecuencias, n_ventanas_temporales)
    """
    padded = np.pad(window, (0, PAD_LENGTH), mode="reflect")
    _, _, Sxx = signal.spectrogram(
        padded,
        fs=1.0,
        window=signal.windows.hann(STFT_CFG["stft_window"]),
        noverlap=STFT_CFG["stft_overlap"],
        nfft=STFT_CFG["nfft"],
        scaling="density",
    )
    # Recortar las columnas del padding
    n_cols_orig = (WINDOW_DAYS - STFT_CFG["stft_window"]) // (
        STFT_CFG["stft_window"] - STFT_CFG["stft_overlap"]
    ) + 1
    return 10 * np.log10(Sxx[:, :n_cols_orig] + 1e-12)


def make_wavelet_image(window: np.ndarray) -> np.ndarray:
    """
    CWT con Wavelet de Morlet compleja (cmor1.5-1.0).
    Resolución adaptativa: ventanas cortas para ciclos cortos,
    ventanas largas para ciclos largos.
    Retorna matriz 2D en dB: (n_scales, n_tiempo)
    Los scales cubren ciclos de 2 a 30 días.
    """
    padded = np.pad(window, (0, PAD_LENGTH), mode="reflect")
    scales = np.array(WAVELET_CFG["scales"], dtype=float)
    coeffs, _ = pywt.cwt(
        padded, scales, WAVELET_CFG["wavelet"], sampling_period=1.0
    )
    power = np.abs(coeffs) ** 2
    # Recortar padding: las primeras WINDOW_DAYS columnas son señal real
    power = power[:, :WINDOW_DAYS]
    return 10 * np.log10(power + 1e-12)


def save_png(matrix: np.ndarray, filepath: str):
    """
    Convierte una matriz 2D a imagen PNG 64×64 usando PIL (sin matplotlib).
    Normaliza a [0,1], aplica colormap magma, flip vertical
    para que frecuencias bajas queden abajo.
    """
    vmin, vmax = matrix.min(), matrix.max()
    norm  = (matrix - vmin) / (vmax - vmin + 1e-10)
    rgba  = cm.get_cmap(CMAP)(norm)
    rgb   = (rgba[:, :, :3] * 255).astype(np.uint8)
    rgb   = rgb[::-1]  # frecuencias bajas abajo
    Image.fromarray(rgb).resize(
        (IMG_SIZE, IMG_SIZE), Image.BILINEAR
    ).save(filepath, format="PNG")


# ═════════════════════════════════════════════════════════════════════════════
# DESCARGA
# ═════════════════════════════════════════════════════════════════════════════

def download(ticker: str, start: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            ticker, start=start, interval="1d",
            progress=False, auto_adjust=True,
        )
        if df is None or len(df) < MIN_HISTORY:
            print(f"  ⚠  {ticker}: {len(df) if df is not None else 0} registros")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Close", "High", "Low"]].dropna()
    except Exception as e:
        print(f"  ✗  {ticker}: {e}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# PROCESAMIENTO DE UN ACTIVO
# ═════════════════════════════════════════════════════════════════════════════

def process_asset(ticker: str, asset_class: str, df: pd.DataFrame,
                  img_fn,             # make_stft_image o make_wavelet_image
                  img_dir_train: str,
                  img_dir_test: str) -> list[dict]:
    """
    Genera imágenes y registros de etiquetas para un activo.
    img_fn es la función de transformación: STFT o Wavelet.
    La misma lógica sirve para los dos datasets.
    """
    close  = df["Close"].values.astype(float)
    high   = df["High"].values.astype(float)
    low    = df["Low"].values.astype(float)
    dates  = df.index

    ret      = log_returns(close)
    ret_norm = zscore(ret)
    atr      = compute_atr(high, low, close, ATR_CFG["period"])

    max_horizon = max(HORIZONS)
    records     = []

    for i in range(WINDOW_DAYS, len(ret_norm) - max_horizon):

        # Validar ATR
        atr_val = atr[i] if i < len(atr) and not np.isnan(atr[i]) else None
        if not atr_val or atr_val < 1e-8:
            continue

        sample_date = dates[i + 1]  # +1 por offset de log_returns
        split       = "test" if sample_date >= pd.Timestamp(TEST_START) else "train"
        img_dir     = img_dir_test if split == "test" else img_dir_train
        sample_id   = f"{ticker}_{i:05d}"
        fname       = f"{sample_id}.png"

        # Generar y guardar imagen
        window   = ret_norm[i - WINDOW_DAYS:i]
        matrix   = img_fn(window)
        save_png(matrix, os.path.join(img_dir, fname))

        # Calcular retornos y etiquetas para todos los horizontes
        record = {
            "sample_id":   sample_id,
            "filename":    fname,
            "ticker":      ticker,
            "asset_class": asset_class,
            "date":        str(sample_date.date()),
            "split":       split,
            "atr":         round(float(atr_val), 6),
        }
        for h in HORIZONS:
            future_ret       = float(np.sum(ret[i:i + h]))
            threshold        = atr_val * ATR_CFG["buy_multiplier"] * np.sqrt(h)
            record[f"ret_{h}d"]   = round(future_ret, 6)
            record[f"label_{h}d"] = make_label(future_ret, threshold)

        records.append(record)

        if len(records) % 200 == 0:
            print(f"     ... {len(records)} muestras", end="\r")

    return records


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE DE GENERACIÓN
# ═════════════════════════════════════════════════════════════════════════════

CSV_FIELDS = (
    ["sample_id", "filename", "ticker", "asset_class", "date", "split", "atr"]
    + [f"ret_{h}d"   for h in HORIZONS]
    + [f"label_{h}d" for h in HORIZONS]
)


def generate(output_dir: str, img_fn, name: str,
             asset_classes: list, start_date: str):
    """
    Genera un dataset completo con la función de imagen indicada.
    name: "STFT" o "Wavelet"  (solo para logging)
    img_fn: make_stft_image o make_wavelet_image
    """
    img_train = os.path.join(output_dir, "images", "train")
    img_test  = os.path.join(output_dir, "images", "test")
    os.makedirs(img_train, exist_ok=True)
    os.makedirs(img_test,  exist_ok=True)

    csv_path = os.path.join(output_dir, "labels.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)

    print("\n" + "═" * 65)
    print(f" DATASET {name}")
    print(f" Output  : {output_dir}")
    print(f" Ventana : {WINDOW_DAYS} días")
    print(f" Train   : {start_date} → {TRAIN_END}")
    print(f" Test    : {TEST_START} → hoy")
    print("═" * 65)

    total        = 0
    label_counts = {h: {"BUY": 0, "HOLD": 0, "SELL": 0}
                    for h in HORIZONS}

    for asset_class in asset_classes:
        tickers = ASSET_UNIVERSE.get(asset_class, [])
        print(f"\n── {asset_class.upper()} ({len(tickers)} activos) ──")

        for ticker in tickers:
            print(f"  📥 {ticker} ...", end=" ", flush=True)
            df = download(ticker, start_date)
            if df is None:
                continue
            print(f"{len(df)} días  "
                  f"{df.index[0].date()} → {df.index[-1].date()}")

            t0      = time.time()
            records = process_asset(
                ticker, asset_class, df,
                img_fn, img_train, img_test,
            )
            elapsed = time.time() - t0

            if not records:
                print("     ⚠  Sin muestras generadas")
                continue

            # Escribir al CSV de forma incremental
            exists = os.path.exists(csv_path)
            with open(csv_path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
                if not exists:
                    writer.writeheader()
                writer.writerows(records)

            total += len(records)
            for r in records:
                for h in HORIZONS:
                    label_counts[h][r[f"label_{h}d"]] += 1

            speed = len(records) / max(elapsed, 0.01)
            print(f"     ✅ {len(records)} muestras  "
                  f"{elapsed:.1f}s  ({speed:.0f} img/s)")

    # ── Resumen ───────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f" {name} — {total} muestras totales")
    print(f"{'─'*65}")

    df_csv = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
    if df_csv is not None:
        tr = len(df_csv[df_csv["split"] == "train"])
        te = len(df_csv[df_csv["split"] == "test"])
        print(f" Train: {tr}   Test: {te}")

    for h in HORIZONS:
        lc = label_counts[h]
        th = max(sum(lc.values()), 1)
        line = f"  {h:2d}d: "
        for lbl in ["BUY", "HOLD", "SELL"]:
            line += f" {lbl} {lc[lbl]:5d} ({lc[lbl]/th*100:.1f}%)"
        print(line)

    # ── Config ────────────────────────────────────────────────────────────
    cfg = {
        "dataset":        name,
        "generated_at":   datetime.now().isoformat(),
        "start_date":     start_date,
        "train_end":      TRAIN_END,
        "test_start":     TEST_START,
        "window_days":    WINDOW_DAYS,
        "img_size":       IMG_SIZE,
        "horizons":       HORIZONS,
        "atr_config":     ATR_CFG,
        "asset_classes":  asset_classes,
        "asset_universe": {k: ASSET_UNIVERSE[k] for k in asset_classes},
        "transform_config": STFT_CFG if name == "STFT" else WAVELET_CFG,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n  labels.csv  →  {csv_path}")
    print(f"  config.json →  {os.path.join(output_dir, 'config.json')}")
    return total


# ═════════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN
# ═════════════════════════════════════════════════════════════════════════════

def verify(output_dir: str):
    csv_path = os.path.join(output_dir, "labels.csv")
    if not os.path.exists(csv_path):
        print(f"✗ No se encontró labels.csv en {output_dir}")
        return

    df = pd.read_csv(csv_path)
    print(f"\n🔍 Verificando {output_dir}")
    print(f"   labels.csv: {len(df)} filas  ·  {len(df.columns)} columnas")

    for split in ["train", "test"]:
        img_path = os.path.join(output_dir, "images", split)
        n_imgs   = len(list(os.scandir(img_path))) if os.path.exists(img_path) else 0
        n_csv    = len(df[df["split"] == split])
        ok       = "✅" if n_imgs == n_csv else "⚠️ "
        print(f"   {ok} {split:5s}: {n_imgs} imágenes  {n_csv} en CSV")

    print(f"\n   Distribución de etiquetas:")
    for h in HORIZONS:
        col = f"label_{h}d"
        if col not in df.columns:
            continue
        counts = df[col].value_counts()
        line   = f"   {h:2d}d:"
        for lbl in ["BUY", "HOLD", "SELL"]:
            n = counts.get(lbl, 0)
            line += f"  {lbl} {n:5d} ({n/len(df)*100:.1f}%)"
        print(line)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera dataset STFT y/o Wavelet para CNN de trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python generate_dataset.py --dataset both
  python generate_dataset.py --dataset stft  --classes crypto fx
  python generate_dataset.py --dataset wavelet --start 2020-01-01
  python generate_dataset.py --verify --out-stft ./data/stft
        """
    )
    parser.add_argument("--dataset",   choices=["stft", "wavelet", "both"],
                        default="both")
    parser.add_argument("--classes",   nargs="+", default=None,
                        choices=list(ASSET_UNIVERSE.keys()))
    parser.add_argument("--out-stft",  default="./dataset_stft")
    parser.add_argument("--out-wavelet", default="./dataset_wavelet")
    parser.add_argument("--start",     default=START_DATE)
    parser.add_argument("--verify",    action="store_true")

    args    = parser.parse_args()
    classes = args.classes or list(ASSET_UNIVERSE.keys())

    if args.verify:
        if args.dataset in ("stft",    "both"): verify(args.out_stft)
        if args.dataset in ("wavelet", "both"): verify(args.out_wavelet)
    else:
        t0 = time.time()
        if args.dataset in ("stft",    "both"):
            generate(args.out_stft,    make_stft_image,    "STFT",
                     classes, args.start)
        if args.dataset in ("wavelet", "both"):
            generate(args.out_wavelet, make_wavelet_image, "Wavelet",
                     classes, args.start)
        print(f"\n⏱  Tiempo total: {(time.time()-t0)/60:.1f} min")