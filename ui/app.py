"""
SpectraTrade — Streamlit demo (modelo real conectado)
=====================================================
Left sidebar  = controles   (mercado · ticker · horizonte · Run)
Main area     = resultados   (precio | espectrograma, luego veredicto + confianza)

Pipeline (idéntico al del último notebook de entrenamiento):
    1. descarga precios (yfinance)
    2. log-returns -> z-score sobre ventana de 90 días
    3. STFT (Hann 32, overlap 28, nfft 64) -> dB -> magma -> flip -> 92x92
    4. modelo STFT entrenado -> BUY / SELL + confianza

Si no encuentra el .pth, cae a un stub aleatorio para que la UI igual demuestre.

Run:  streamlit run ui/app.py
"""

import io
import os
import sys
import numpy as np
import streamlit as st
from scipy import signal
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import pywt
    PYWT_OK = True
except ImportError:
    PYWT_OK = False

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_OK = True
    TORCH_ERROR = None
except ImportError:
    TORCH_OK = False
    TORCH_ERROR = str(sys.exc_info()[1])


# ── configuración del modelo (debe coincidir con el entrenamiento) ────────────
CLASSES     = ["BUY", "SELL"]
WINDOW_DAYS = 90
IMG_SIZE    = 92
CMAP        = "magma"
PAD_LENGTH  = 16
STFT_CFG    = {"stft_window": 32, "stft_overlap": 28, "nfft": 64}
WAVELET_CFG = {"wavelet": "cmor1.5-1.0", "scales": list(range(2, 31))}

_HERE       = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_HERE, "..", "models")

# modelos disponibles en la UI: nombre -> (archivo, representación, arquitectura)
MODEL_OPTIONS = {
    "CNN v1 · STFT":          ("cnn_best_v1.pth", "STFT", "cnn_v1"),
    "CNN v2 · STFT":          ("cnn_best_v2.pth", "STFT", "cnn_v2"),
    "EfficientNet B0 · STFT": ("efficientnet_best.pth", "STFT", "efficientnet_b0"),
    "ResNet18 · STFT":        ("resnet_best.pth", "STFT", "resnet18"),
}


# ── page + theme ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="SpectraTrade", page_icon="✶", layout="wide")

ACCENT = "#2a93a8"
UP, DOWN = "#2e9e6b", "#d65a45"
CLR = {"BUY": UP, "SELL": DOWN}

st.markdown(f"""
<style>
  .block-container {{ padding-top: 2rem; }}
  section[data-testid="stSidebar"] {{ background: #f3efe6; }}
  .card {{ border: 1px solid #e4ded2; border-radius: 16px; padding: 18px 20px;
           background: #fbfaf5; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
  .mono {{ font-family: ui-monospace, monospace; font-size: 11px;
           letter-spacing: .06em; text-transform: uppercase; color: #7a766d; }}
  .verdict {{ font-size: 44px; font-weight: 800; line-height: 1; }}
  .arrow {{ font-size: 60px; line-height: 1; }}

  /* ── sidebar: paleta unificada sobre crema con acento teal ───────────── */
  /* radio (modelo) */
  section[data-testid="stSidebar"] [role="radiogroup"] label p {{
      color: #3a3730; font-weight: 600; }}
  section[data-testid="stSidebar"] [role="radiogroup"] [data-baseweb="radio"] div:first-child {{
      border-color: {ACCENT}; }}

  /* pills (market) y segmented_control (horizonte): mismo look pill teal */
  section[data-testid="stSidebar"] button[kind="pills"],
  section[data-testid="stSidebar"] button[kind="pillsActive"],
  section[data-testid="stSidebar"] button[kind="segmentedControl"],
  section[data-testid="stSidebar"] button[kind="segmentedControlActive"] {{
      border-radius: 999px; font-weight: 600; border: 1px solid {ACCENT}55;
      background: #fbfaf5; color: #3a3730; transition: all .12s ease; }}
  section[data-testid="stSidebar"] button[kind="pills"]:hover,
  section[data-testid="stSidebar"] button[kind="segmentedControl"]:hover {{
      border-color: {ACCENT}; color: {ACCENT}; background: {ACCENT}11; }}
  /* estado seleccionado: relleno teal sólido */
  section[data-testid="stSidebar"] button[kind="pillsActive"],
  section[data-testid="stSidebar"] button[kind="segmentedControlActive"] {{
      background: {ACCENT}; border-color: {ACCENT}; color: #ffffff; }}
  section[data-testid="stSidebar"] button[kind="pillsActive"] p,
  section[data-testid="stSidebar"] button[kind="segmentedControlActive"] p {{
      color: #ffffff; }}

  /* selectbox (ticker): borde teal, fondo claro, texto negro */
  section[data-testid="stSidebar"] [data-baseweb="select"] > div {{
      background: #fbfaf5; border-color: {ACCENT}55; border-radius: 10px; }}
  section[data-testid="stSidebar"] [data-baseweb="select"] div {{
      color: #000000; }}

  /* botón Run: acento teal en vez del coral por defecto */
  .stButton>button {{ border-radius: 12px; font-weight: 700; }}
  section[data-testid="stSidebar"] .stButton>button[kind="primary"] {{
      background: {ACCENT}; border-color: {ACCENT}; color: #ffffff; }}
  section[data-testid="stSidebar"] .stButton>button[kind="primary"]:hover {{
      background: #237c8e; border-color: #237c8e; }}
</style>
""", unsafe_allow_html=True)


# ── modelos: arquitecturas entrenadas en el notebook ─────────────────────────
if TORCH_OK:
    class SpectrogramCNN(nn.Module):
        def __init__(self, num_classes=3, dropout=0.3):
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
                conv_block(3, 32, pool=True, drop=0.1),
                conv_block(32, 64, pool=True, drop=0.15),
                conv_block(64, 128, pool=True, drop=0.2),
                conv_block(128, 256, pool=True, drop=0.25),
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
        def __init__(self, num_classes=3, dropout=0.4):
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
                conv_block(3, 32, pool=True, drop=0.1),
                conv_block(32, 64, pool=True, drop=0.15),
                conv_block(64, 128, pool=True, drop=0.2),
                conv_block(128, 256, pool=True, drop=0.25),
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
        def __init__(self, num_classes=3):
            super().__init__()
            base = models.resnet18(weights=None)
            in_features = base.fc.in_features
            base.fc = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, num_classes),
            )
            self.model = base

        def forward(self, x):
            return self.model(x)


    class EfficientNetFinancial(nn.Module):
        def __init__(self, num_classes=3):
            super().__init__()
            base = models.efficientnet_b0(weights=None)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes),
            )
            self.model = base

        def forward(self, x):
            return self.model(x)


    def _build_model(architecture: str, num_classes=3):
        builders = {
            "cnn_v1": SpectrogramCNN,
            "cnn_v2": SpectrogramCNNv2,
            "resnet18": ResNetFinancial,
            "efficientnet_b0": EfficientNetFinancial,
        }
        return builders[architecture](num_classes=num_classes)

else:
    def _build_model(architecture: str, num_classes=3):
        raise RuntimeError("PyTorch/torchvision no están disponibles")


@st.cache_resource(show_spinner=False)
def load_model(filename: str, architecture: str):
    """Carga un checkpoint por nombre de archivo y arquitectura. Devuelve (model, ok, reason)."""
    path = os.path.join(_MODELS_DIR, filename)
    if not TORCH_OK:
        return None, False, "PyTorch/torchvision no están disponibles en este entorno"
    if not os.path.exists(path):
        return None, False, f"No existe el archivo: {path}"
    try:
        model = _build_model(architecture, len(CLASSES))
        state = torch.load(path, map_location="cpu")
        has_wrapper_prefix = any(k.startswith("model.") for k in state)

        if hasattr(model, "model"):
            target = model.model
            if has_wrapper_prefix:
                state = {
                    k.replace("model.", "", 1) if k.startswith("model.") else k: v
                    for k, v in state.items()
                }
        else:
            target = model

        target.load_state_dict(state, strict=True)
        model.eval()
        return model, True, None
    except Exception as e:
        return None, False, str(e)


_EVAL_TF = (transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
]) if TORCH_OK else None)


# ── helpers ───────────────────────────────────────────────────────────────────
PERIOD_MAP = {  # horizonte de predicción -> días de descarga
    "15d": "next 15 days",
}

@st.cache_data(show_spinner=False)
def load_prices_with_dates(ticker: str):
    """Descarga histórico amplio (1+ años). Retorna (closes, dates)."""
    if yf is not None and ticker:
        try:
            df = yf.download(ticker, period="2y", interval="1d",
                             progress=False, auto_adjust=True)
            closes = df["Close"].dropna().to_numpy().ravel()
            dates = df["Close"].dropna().index.to_numpy()
            if closes.size >= WINDOW_DAYS + 1:
                return closes, dates
        except Exception:
            pass
    # fallback: random walk con 500 días
    rng = np.random.default_rng(abs(hash(ticker)) % 2**32)
    steps = rng.normal(0, 1, 500).cumsum()
    prices = 100 + steps - steps.min()
    dates = np.array([datetime.now() - timedelta(days=500-i) for i in range(500)])
    return prices, dates


@st.cache_data(show_spinner=False)
def load_prices(ticker: str) -> np.ndarray:
    """Compat. Retorna solo los precios."""
    closes, _ = load_prices_with_dates(ticker)
    return closes


def make_stft_matrix(prices: np.ndarray) -> np.ndarray:
    """Replica make_stft_image() del notebook: z-score de log-returns -> STFT dB."""
    returns = np.diff(np.log(prices + 1e-9))
    window  = returns[-WINDOW_DAYS:]
    std = window.std()
    window = (window - window.mean()) / std if std > 1e-10 else np.zeros_like(window)

    padded = np.pad(window, (0, PAD_LENGTH), mode="reflect")
    _, _, Sxx = signal.spectrogram(
        padded, fs=1.0,
        window=signal.windows.hann(STFT_CFG["stft_window"]),
        noverlap=STFT_CFG["stft_overlap"],
        nfft=STFT_CFG["nfft"], scaling="density",
    )
    n_cols = (WINDOW_DAYS - STFT_CFG["stft_window"]) // (
        STFT_CFG["stft_window"] - STFT_CFG["stft_overlap"]) + 1
    return 10 * np.log10(Sxx[:, :n_cols] + 1e-12)


def make_wavelet_matrix(prices: np.ndarray) -> np.ndarray:
    """Replica make_wavelet_image(): CWT Morlet (cmor1.5-1.0) sobre z-score -> dB."""
    returns = np.diff(np.log(prices + 1e-9))
    window  = returns[-WINDOW_DAYS:]
    std = window.std()
    window = (window - window.mean()) / std if std > 1e-10 else np.zeros_like(window)

    padded = np.pad(window, (0, PAD_LENGTH), mode="reflect")
    scales = np.array(WAVELET_CFG["scales"], dtype=float)
    coeffs, _ = pywt.cwt(padded, scales, WAVELET_CFG["wavelet"], sampling_period=1.0)
    power = np.abs(coeffs) ** 2
    power = power[:, :WINDOW_DAYS]
    return 10 * np.log10(power + 1e-12)


def make_matrix(prices: np.ndarray, repr_name: str) -> np.ndarray:
    if repr_name == "Wavelet":
        return make_wavelet_matrix(prices)
    return make_stft_matrix(prices)


def matrix_to_png(matrix: np.ndarray) -> Image.Image:
    """Misma conversión que save_png(): normaliza, magma, flip vertical, 92x92."""
    vmin, vmax = matrix.min(), matrix.max()
    norm = (matrix - vmin) / (vmax - vmin + 1e-10)
    rgb  = (cm.get_cmap(CMAP)(norm)[:, :, :3] * 255).astype(np.uint8)[::-1]
    return Image.fromarray(rgb).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)


def predict(prices: np.ndarray, model, model_ok: bool, repr_name: str):
    """Devuelve (label, confianza, probs_dict, imagen_PIL)."""
    matrix = make_matrix(prices, repr_name)
    img    = matrix_to_png(matrix)

    if model_ok and model is not None:
        x = _EVAL_TF(img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).numpy()[0]
        idx = int(probs.argmax())
        return CLASSES[idx], float(probs[idx]), dict(zip(CLASSES, probs.tolist())), img

    # fallback stub
    rng = np.random.default_rng(int(abs(matrix.sum())) % 2**32)
    p = rng.dirichlet([1, 1])
    idx = int(p.argmax())
    return CLASSES[idx], float(p[idx]), dict(zip(CLASSES, p.tolist())), img


# ── ticker options by market ──────────────────────────────────────────────────
TICKER_MAP = {
    "Crypto": {
        "BTC — Bitcoin": "BTC-USD", "ETH — Ethereum": "ETH-USD",
        "SOL — Solana": "SOL-USD",
    },
    "Stocks": {
        "AAPL — Apple": "AAPL", "MSFT — Microsoft": "MSFT",
        "GOOGL — Alphabet": "GOOGL", "AMZN — Amazon": "AMZN",
        "NVDA — Nvidia": "NVDA", "META — Meta": "META",
        "TSLA — Tesla": "TSLA", "SPY — S&P 500 ETF": "SPY",
    },
    "Commodities": {
        "GC — Gold": "GC=F", "SI — Silver": "SI=F",
    },
    "FX": {
        "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X",
    },
}

# ── sidebar = controles ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="mono">modelo</div>', unsafe_allow_html=True)
    model_label = st.radio("Modelo", list(MODEL_OPTIONS.keys()),
                           label_visibility="collapsed")
    model_file, repr_name, architecture = MODEL_OPTIONS[model_label]
    model, model_ok, model_error = load_model(model_file, architecture)
    if model_ok:
        st.success(f"{model_label} cargado ✓")
    else:
        st.error("Stub (sin modelo)")
        st.caption(model_error)
        st.caption(f"Python activo: {sys.executable}")
        if TORCH_ERROR:
            st.caption(f"Detalle import torch: {TORCH_ERROR}")

    st.markdown('<div class="mono">market</div>', unsafe_allow_html=True)
    market = st.pills("Market", list(TICKER_MAP.keys()),
                      default="Crypto", label_visibility="collapsed") or "Crypto"

    st.markdown('<div class="mono">ticker</div>', unsafe_allow_html=True)
    options   = list(TICKER_MAP[market].keys())
    selection = st.selectbox("Ticker", options, label_visibility="collapsed")
    ticker    = TICKER_MAP[market][selection]

    st.markdown('<div class="mono">horizonte de predicción</div>', unsafe_allow_html=True)
    period_key = st.segmented_control("Horizonte", list(PERIOD_MAP.keys()),
                                      default="15d", label_visibility="collapsed") or "15d"

    st.markdown('<div class="mono">test retrospectivo</div>', unsafe_allow_html=True)
    test_mode = st.checkbox("Simular predicción en fecha pasada", value=False,
                            label_visibility="collapsed")
    
    test_date = None
    if test_mode:
        test_date = st.date_input("Fecha de test", value=datetime.now() - timedelta(days=60),
                                  label_visibility="collapsed")

    run = st.button("⚡ Run prediction", type="primary", use_container_width=True)


# ── main area = resultados ────────────────────────────────────────────────────
horizon = PERIOD_MAP[period_key]
horizon_days = int(period_key.replace("d", ""))

# La predicción solo se dispara al apretar "Run prediction"; el resultado se
# guarda en session_state para sobrevivir a los reruns de Streamlit.
if run:
    closes, dates = load_prices_with_dates(ticker)
    
    if test_mode and test_date:
        # ─── Test retrospectivo ──────────────────────────────────────────
        test_date_np = np.datetime64(test_date)
        date_idx = np.where(dates == test_date_np)[0]
        
        if len(date_idx) == 0:
            # Si no hay fecha exacta, buscar el más cercano
            date_idx = np.argmin(np.abs(dates - test_date_np))
        else:
            date_idx = date_idx[0]
        
        # Validar que hay suficientes datos
        if date_idx < WINDOW_DAYS + 1:
            st.error(f"❌ Necesitamos al menos {WINDOW_DAYS} días previos a la fecha de test")
            st.stop()
        
        if date_idx + horizon_days > len(closes):
            st.error(f"❌ Necesitamos al menos {horizon_days} días de datos después de la fecha de test")
            st.stop()
        
        # Extraer precios: 90 días previos para hacer la predicción
        prices_for_pred = closes[date_idx - WINDOW_DAYS:date_idx]
        label, conf, probs, img = predict(prices_for_pred, model, model_ok, repr_name)
        
        # Datos para visualización:
        # - histórico: últimos 90 días antes de test_date
        # - predicción: los días que marca el horizonte
        # - realidad: los datos reales de ese horizonte
        hist_prices = closes[date_idx - WINDOW_DAYS:date_idx]
        real_prices = closes[date_idx:date_idx + horizon_days]
        hist_dates = dates[date_idx - WINDOW_DAYS:date_idx]
        real_dates = dates[date_idx:date_idx + horizon_days]
        
        st.session_state["result"] = {
            "selection": selection, "horizon": horizon, "prices": prices_for_pred,
            "label": label, "conf": conf, "probs": probs, "img": img,
            "test_mode": True,
            "test_date": test_date,
            "hist_prices": hist_prices,
            "hist_dates": hist_dates,
            "real_prices": real_prices,
            "real_dates": real_dates,
        }
    else:
        # ─── Predicción normal (último dato) ──────────────────────────────
        prices = closes[-WINDOW_DAYS:]
        label, conf, probs, img = predict(closes, model, model_ok, repr_name)
        st.session_state["result"] = {
            "selection": selection, "horizon": horizon, "prices": prices,
            "label": label, "conf": conf, "probs": probs, "img": img,
            "test_mode": False,
        }

if "result" not in st.session_state:
    st.info("Elegí mercado, ticker y horizonte, luego apretá **⚡ Run prediction**.")
    st.stop()

res       = st.session_state["result"]
selection = res["selection"]
horizon   = res["horizon"]
prices    = res["prices"]
label, conf, probs, img = res["label"], res["conf"], res["probs"], res["img"]
test_mode = res.get("test_mode", False)

st.markdown(f"### {selection}  ·  <span class='mono'>{horizon}</span>",
            unsafe_allow_html=True)

col_chart, col_spec = st.columns([0.55, 0.45])
with col_chart:
    if test_mode:
        # ─── Test retrospectivo: histórico + predicción + realidad ─────────
        st.markdown('<div class="mono">análisis retrospectivo</div>',
                    unsafe_allow_html=True)
        
        hist_prices = res["hist_prices"]
        real_prices = res["real_prices"]
        test_date = res["test_date"]
        
        # Crear figura con matplotlib para más control
        fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
        
        # Histórico: línea gris
        ax.plot(range(len(hist_prices)), hist_prices, 
                label="Histórico (90d)", color="#999999", linewidth=1.5, alpha=0.7)
        
        # Realidad: línea azul/verde (coloreada por si subió o bajó)
        real_color = UP if real_prices[-1] > hist_prices[-1] else DOWN
        ax.plot(range(len(hist_prices), len(hist_prices) + len(real_prices)), real_prices,
            label=f"Realidad ({horizon_days}d)", color=real_color, linewidth=2, marker="o", markersize=3)
        
        # Línea vertical separadora
        ax.axvline(x=len(hist_prices) - 0.5, color="#ccc", linestyle="--", linewidth=1, alpha=0.5)
        
        # Predicción (hipotética): proyección simple del último valor
        # Si el modelo predijo BUY, asumimos subida; SELL, asumimos bajada
        last_price = hist_prices[-1]
        trend = 0.05 if label == "BUY" else -0.05
        pred_prices = np.linspace(last_price, last_price * (1 + trend), len(real_prices))
        pred_color = UP if label == "BUY" else DOWN
        ax.plot(range(len(hist_prices), len(hist_prices) + len(real_prices)), pred_prices,
                label=f"Predicción ({label})", color=pred_color, linewidth=2, 
                linestyle=":", alpha=0.7)
        
        # Etiquetas y formato
        ax.set_xlabel("Días", fontsize=9, color="#666")
        ax.set_ylabel("Precio", fontsize=9, color="#666")
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.2)
        ax.set_facecolor("#f8f8f8")
        fig.patch.set_facecolor("#ffffff")
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Acierto/error de la predicción
        actual_direction = "📈 Subió" if real_prices[-1] > hist_prices[-1] else "📉 Bajó"
        pred_match = "✓ Correcto" if (label == "BUY" and real_prices[-1] > hist_prices[-1]) or \
                                    (label == "SELL" and real_prices[-1] < hist_prices[-1]) else "✗ Incorrecto"
        st.caption(f"Predicción: **{label}** | Resultado real: **{actual_direction}** | {pred_match}")
    else:
        # ─── Predicción normal (últimos 90 días) ──────────────────────────
        st.markdown('<div class="mono">price history (últimos 90 días)</div>',
                    unsafe_allow_html=True)
        st.line_chart(prices[-WINDOW_DAYS:], height=240, color=ACCENT)

with col_spec:
    rep_lbl = "escalograma Wavelet" if repr_name == "Wavelet" else "espectrograma STFT"
    st.markdown(f'<div class="mono">{rep_lbl} · input del modelo</div>',
                unsafe_allow_html=True)
    st.image(img, width="stretch")

st.write("")
color = CLR[label]
arrow = {"BUY": "↑", "SELL": "↓"}[label]
txt   = {"BUY": "BUY", "SELL": "SELL"}[label]

v_left, v_right = st.columns([0.6, 0.4])
with v_left:
    st.markdown(f"""
      <div class="card" style="background:{color}14;border-color:{color}55;
           display:flex;gap:18px;align-items:center;">
        <div class="arrow" style="color:{color}">{arrow}</div>
        <div>
          <div class="mono">señal · {horizon}</div>
          <div class="verdict" style="color:{color}">{txt}</div>
        </div>
      </div>
    """, unsafe_allow_html=True)
with v_right:
    st.markdown('<div class="mono">confianza</div>', unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:40px;font-weight:800;color:{color}'>"
                f"{conf*100:.0f}%</div>", unsafe_allow_html=True)
    st.progress(conf)

# desglose por clase
st.write("")
st.markdown('<div class="mono">probabilidades del modelo</div>', unsafe_allow_html=True)
buy_col, sell_col = st.columns(2)
with buy_col:
    st.metric("BUY", f"{probs['BUY']*100:.1f}%")
with sell_col:
    st.metric("SELL", f"{probs['SELL']*100:.1f}%")
