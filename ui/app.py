"""
SpectraTrade — option B (split layout) in Streamlit
===================================================
Left sidebar  = controls rail   (ticker search · market · period · Run)
Main area     = results          (price chart | spectrogram, then verdict + confidence)

This is a working scaffold:
  • fetches price history (yfinance; falls back to a synthetic random walk offline)
  • computes a frequency spectrogram from the price series (scipy)
  • renders that spectrogram as the image you feed your CV model
  • calls predict_from_spectrogram(...) — plug your model in there

Run:  streamlit run app.py
"""

import io
import numpy as np
import streamlit as st
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except ImportError:
    yf = None


# ── page + theme ────────────────────────────────────────────────────────────
st.set_page_config(page_title="SpectraTrade", page_icon="✶", layout="wide")

ACCENT = "#2a93a8"
UP, DOWN, HOLD = "#2e9e6b", "#d65a45", "#caa23c"

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


# ── helpers ─────────────────────────────────────────────────────────────────
PERIOD_MAP = {  # preset -> (yf period, yf interval, human horizon)
    "1D": ("1d", "5m", "next 1 day"),
    "1W": ("5d", "30m", "next 1 week"),
    "1M": ("1mo", "1h", "next 1 month"),
}

@st.cache_data(show_spinner=False)
def load_prices(ticker: str, period: str, interval: str) -> np.ndarray:
    """Return a 1-D close-price array. Falls back to a synthetic series offline."""
    if yf is not None and ticker:
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             progress=False, auto_adjust=True)
            closes = df["Close"].dropna().to_numpy().ravel()
            if closes.size >= 32:
                return closes
        except Exception:
            pass
    # offline / failed fetch → deterministic random walk so the UI still demos
    rng = np.random.default_rng(abs(hash((ticker, period))) % 2**32)
    steps = rng.normal(0, 1, 240).cumsum()
    return 100 + steps - steps.min()


def make_spectrogram(prices: np.ndarray):
    """Compute a spectrogram of price returns and render it to a PNG (the CV input)."""
    returns = np.diff(np.log(prices + 1e-9))
    nper = max(16, min(64, returns.size // 4))
    f, t, Sxx = signal.spectrogram(returns, nperseg=nper, noverlap=nper // 2)
    Sxx = 10 * np.log10(Sxx + 1e-12)

    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=110)
    ax.pcolormesh(t, f, Sxx, shading="gouraud", cmap="magma")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf, Sxx


def predict_from_spectrogram(spectrogram_array: np.ndarray, ticker: str):
    """
    >>> PLUG YOUR CV MODEL HERE <<<
    Take the spectrogram image/array, return (label, confidence_0_to_1).

        model = load_model(...)
        logits = model(preprocess(spectrogram_array))
        idx = logits.argmax(); return CLASSES[idx], float(softmax(logits)[idx])

    The stub below is deterministic so the app runs end-to-end without a model.
    """
    rng = np.random.default_rng(abs(hash(ticker)) % 2**32)
    label = rng.choice(["UP", "DOWN", "HOLD"], p=[0.45, 0.3, 0.25])
    conf = float(rng.uniform(0.55, 0.85))
    return label, conf


# ── ticker options by market ─────────────────────────────────────────────────
TICKER_MAP = {
    "Stocks": {
        "AAPL — Apple": "AAPL",
        "MSFT — Microsoft": "MSFT",
        "GOOGL — Alphabet (Google)": "GOOGL",
        "AMZN — Amazon": "AMZN",
        "NVDA — Nvidia": "NVDA",
        "META — Meta (Facebook)": "META",
        "TSLA — Tesla": "TSLA",
        "JPM — JPMorgan Chase": "JPM",
        "V — Visa": "V",
        "JNJ — Johnson & Johnson": "JNJ",
        "WMT — Walmart": "WMT",
        "PG — Procter & Gamble": "PG",
        "XOM — ExxonMobil": "XOM",
        "BAC — Bank of America": "BAC",
        "DIS — Disney": "DIS",
    },
    "Crypto": {
        "BTC — Bitcoin": "BTC-USD",
        "ETH — Ethereum": "ETH-USD",
        "BNB — Binance Coin": "BNB-USD",
        "SOL — Solana": "SOL-USD",
        "XRP — Ripple": "XRP-USD",
        "ADA — Cardano": "ADA-USD",
        "DOGE — Dogecoin": "DOGE-USD",
        "AVAX — Avalanche": "AVAX-USD",
        "DOT — Polkadot": "DOT-USD",
        "MATIC — Polygon": "MATIC-USD",
    },
    "FX": {
        "EUR/USD — Euro / US Dollar": "EURUSD=X",
        "GBP/USD — British Pound / US Dollar": "GBPUSD=X",
        "USD/JPY — US Dollar / Japanese Yen": "USDJPY=X",
        "AUD/USD — Australian Dollar / US Dollar": "AUDUSD=X",
        "USD/CAD — US Dollar / Canadian Dollar": "USDCAD=X",
        "USD/CHF — US Dollar / Swiss Franc": "USDCHF=X",
        "NZD/USD — New Zealand Dollar / US Dollar": "NZDUSD=X",
        "EUR/GBP — Euro / British Pound": "EURGBP=X",
        "EUR/JPY — Euro / Japanese Yen": "EURJPY=X",
        "GBP/JPY — British Pound / Japanese Yen": "GBPJPY=X",
    },
    "Commodities": {
        "GC — Gold": "GC=F",
        "SI — Silver": "SI=F",
        "CL — Crude Oil": "CL=F",
        "NG — Natural Gas": "NG=F",
        "HG — Copper": "HG=F",
        "PL — Platinum": "PL=F",
        "PA — Palladium": "PA=F",
        "ZC — Corn": "ZC=F",
        "ZW — Wheat": "ZW=F",
        "ZS — Soybeans": "ZS=F",
    },
}

# ── sidebar = controls ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="mono">market</div>', unsafe_allow_html=True)
    market = st.pills("Market", ["Stocks", "Crypto", "FX", "Commodities"],
                      default="Crypto", label_visibility="collapsed")

    st.markdown('<div class="mono">ticker</div>', unsafe_allow_html=True)
    market = market or "Crypto"
    options = list(TICKER_MAP[market].keys())
    selection = st.selectbox("Ticker", options, label_visibility="collapsed")
    ticker = TICKER_MAP[market][selection]

    st.markdown('<div class="mono">prediction period</div>', unsafe_allow_html=True)
    period_key = st.segmented_control("Period", list(PERIOD_MAP.keys()),
                                      default="1W", label_visibility="collapsed")

    run = st.button("⚡ Run prediction", type="primary", use_container_width=True)


# ── main area = results ─────────────────────────────────────────────────────
period_key = period_key or "1W"
yf_period, yf_interval, horizon = PERIOD_MAP[period_key]
prices = load_prices(ticker, yf_period, yf_interval)
spec_png, spec_arr = make_spectrogram(prices)
label, conf = predict_from_spectrogram(spec_arr, ticker or "—")

st.markdown(f"### {selection}  ·  <span class='mono'>{horizon}</span>", unsafe_allow_html=True)

col_chart, col_spec = st.columns([0.55, 0.45])
with col_chart:
    st.markdown('<div class="mono">price history</div>', unsafe_allow_html=True)
    st.line_chart(prices, height=240, color=ACCENT)
with col_spec:
    st.markdown('<div class="mono">spectrogram · CV model input</div>', unsafe_allow_html=True)
    st.image(spec_png, width="stretch")

st.write("")
color = {"UP": UP, "DOWN": DOWN, "HOLD": HOLD}[label]
arrow = {"UP": "↑", "DOWN": "↓", "HOLD": "→"}[label]
txt = {"UP": "LIKELY UP", "DOWN": "LIKELY DOWN", "HOLD": "HOLD"}[label]

v_left, v_right = st.columns([0.6, 0.4])
with v_left:
    st.markdown(f"""
      <div class="card" style="background:{color}14;border-color:{color}55;
           display:flex;gap:18px;align-items:center;">
        <div class="arrow" style="color:{color}">{arrow}</div>
        <div>
          <div class="mono">prediction · {horizon}</div>
          <div class="verdict" style="color:{color}">{txt}</div>
        </div>
      </div>
    """, unsafe_allow_html=True)
with v_right:
    st.markdown('<div class="mono">confidence</div>', unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:40px;font-weight:800;color:{color}'>{conf*100:.0f}%</div>",
                unsafe_allow_html=True)
    st.progress(conf)
