"""
SpectraTrade — Streamlit demo (modelo real conectado)
=====================================================
Left sidebar  = controles   (mercado · ticker · horizonte · Run)
Main area     = resultados   (precio | espectrograma, luego veredicto + confianza)

Pipeline (idéntico al de entrenamiento, ver proyecto_final_resultados.ipynb):
  1. descarga precios (yfinance)
  2. log-returns -> z-score sobre ventana de 60 días
  3. STFT (Hann 32, overlap 28, nfft 64) -> dB -> magma -> flip -> 64x64
  4. ResNet-STFT entrenado -> BUY / HOLD / SELL + confianza

Si no encuentra el .pth, cae a un stub aleatorio para que la UI igual demuestre.

Run:  streamlit run ui/app.py
"""

import io
import os
import numpy as np
import streamlit as st
from scipy import signal
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
except ImportError:
    TORCH_OK = False


# ── configuración del modelo (debe coincidir con el entrenamiento) ────────────
CLASSES     = ["BUY", "HOLD", "SELL"]      # orden alfabético, igual que en el notebook
WINDOW_DAYS = 60
IMG_SIZE    = 64
CMAP        = "magma"
PAD_LENGTH  = 16
STFT_CFG    = {"stft_window": 32, "stft_overlap": 28, "nfft": 64}
WAVELET_CFG = {"wavelet": "cmor1.5-1.0", "scales": list(range(2, 31))}

_HERE       = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_HERE, "..", "models")

# modelos disponibles en la UI: nombre -> (archivo, representación)
MODEL_OPTIONS = {
    "ResNet · STFT":    ("model_RESNET-STFT.pth",    "STFT"),
    "ResNet · Wavelet": ("model_RESNET-Wavelet.pth", "Wavelet"),
}


# ── page + theme ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="SpectraTrade", page_icon="✶", layout="wide")

ACCENT = "#2a93a8"
UP, DOWN, HOLD = "#2e9e6b", "#d65a45", "#caa23c"   # BUY, SELL, HOLD
CLR = {"BUY": UP, "SELL": DOWN, "HOLD": HOLD}

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
  .stButton>button {{ border-radius: 12px; font-weight: 700; }}
</style>
""", unsafe_allow_html=True)


# ── modelo: ResNet18 con cabeza custom (idéntica a ResNetFinancial) ───────────
def _build_resnet(num_classes=3):
    base = models.resnet18(weights=None)
    in_f = base.fc.in_features
    base.fc = nn.Sequential(
        nn.Linear(in_f, 128), nn.ReLU(), nn.Dropout(0.55),
        nn.Linear(128, num_classes),
    )
    return base


@st.cache_resource(show_spinner=False)
def load_model(filename: str):
    """Carga un ResNet entrenado por nombre de archivo. Devuelve (model, ok)."""
    path = os.path.join(_MODELS_DIR, filename)
    if not TORCH_OK or not os.path.exists(path):
        return None, False
    try:
        model = _build_resnet(len(CLASSES))
        state = torch.load(path, map_location="cpu")
        # run_experiment guardó base.fc bajo el wrapper ResNetFinancial -> 'model.'
        state = {k.replace("model.", "", 1) if k.startswith("model.") else k: v
                 for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        model.eval()
        return model, True
    except Exception as e:
        st.sidebar.warning(f"No se pudo cargar el modelo: {e}")
        return None, False


_EVAL_TF = (transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
]) if TORCH_OK else None)


# ── helpers ───────────────────────────────────────────────────────────────────
PERIOD_MAP = {  # horizonte de predicción -> días de descarga
    "7d":  "next 7 days",
    "15d": "next 15 days",
    "30d": "next 30 days",
}

@st.cache_data(show_spinner=False)
def load_prices(ticker: str) -> np.ndarray:
    """Cierres diarios (>= 90 días). Cae a random walk si yfinance falla."""
    if yf is not None and ticker:
        try:
            df = yf.download(ticker, period="6mo", interval="1d",
                             progress=False, auto_adjust=True)
            closes = df["Close"].dropna().to_numpy().ravel()
            if closes.size >= WINDOW_DAYS + 1:
                return closes
        except Exception:
            pass
    rng = np.random.default_rng(abs(hash(ticker)) % 2**32)
    steps = rng.normal(0, 1, 240).cumsum()
    return 100 + steps - steps.min()


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
    """Misma conversión que save_png(): normaliza, magma, flip vertical, 64x64."""
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
    p = rng.dirichlet([1, 1, 1])
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
    model_file, repr_name = MODEL_OPTIONS[model_label]
    model, model_ok = load_model(model_file)
    if model_ok:
        st.success(f"{model_label} cargado ✓")
    else:
        st.error("Stub (sin modelo)")
        st.caption(f"Falta `models/{model_file}`.")

    st.markdown('<div class="mono">market</div>', unsafe_allow_html=True)
    market = st.pills("Market", list(TICKER_MAP.keys()),
                      default="Crypto", label_visibility="collapsed") or "Crypto"

    st.markdown('<div class="mono">ticker</div>', unsafe_allow_html=True)
    options   = list(TICKER_MAP[market].keys())
    selection = st.selectbox("Ticker", options, label_visibility="collapsed")
    ticker    = TICKER_MAP[market][selection]

    st.markdown('<div class="mono">horizonte de predicción</div>', unsafe_allow_html=True)
    period_key = st.segmented_control("Horizonte", list(PERIOD_MAP.keys()),
                                      default="7d", label_visibility="collapsed") or "7d"

    run = st.button("⚡ Run prediction", type="primary", use_container_width=True)


# ── main area = resultados ────────────────────────────────────────────────────
horizon = PERIOD_MAP[period_key]
prices  = load_prices(ticker)
label, conf, probs, img = predict(prices, model, model_ok, repr_name)

st.markdown(f"### {selection}  ·  <span class='mono'>{horizon}</span>",
            unsafe_allow_html=True)

col_chart, col_spec = st.columns([0.55, 0.45])
with col_chart:
    st.markdown('<div class="mono">price history (últimos 60 días)</div>',
                unsafe_allow_html=True)
    st.line_chart(prices[-WINDOW_DAYS:], height=240, color=ACCENT)
with col_spec:
    rep_lbl = "escalograma Wavelet" if repr_name == "Wavelet" else "espectrograma STFT"
    st.markdown(f'<div class="mono">{rep_lbl} · input del modelo</div>',
                unsafe_allow_html=True)
    st.image(img, width="stretch")

st.write("")
color = CLR[label]
arrow = {"BUY": "↑", "SELL": "↓", "HOLD": "→"}[label]
txt   = {"BUY": "BUY", "SELL": "SELL", "HOLD": "HOLD"}[label]

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
st.markdown('<div class="mono">probabilidades por clase</div>', unsafe_allow_html=True)
pc = st.columns(len(CLASSES))
for c, cls in zip(pc, CLASSES):
    with c:
        st.metric(cls, f"{probs[cls]*100:.1f}%")
