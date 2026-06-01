# SpectraTrade — Streamlit scaffold (option B layout)

Run it:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it maps to wireframe option B
| Wireframe B            | Streamlit                                   |
|------------------------|---------------------------------------------|
| Left controls rail     | `st.sidebar` (search · market · period · Run) |
| Results: chart + spectrogram side by side | `st.columns([0.55, 0.45])` |
| Big verdict + confidence | HTML card via `st.markdown` + `st.progress` |

## Plug in your model
Replace the body of `predict_from_spectrogram(spectrogram_array, ticker)` in `app.py`.
It already receives the same spectrogram that's shown on screen, so the image your
CV model sees is exactly what the user sees.

## Notes
- Price data comes from `yfinance` (handles stocks, `BTC-USD` crypto, `EURUSD=X` FX).
  Offline / failed fetches fall back to a synthetic series so the UI always renders.
- `st.pills` / `st.segmented_control` need **Streamlit ≥ 1.40**.
- The sketchy hand-drawn wireframe look is intentionally dropped — this uses native
  Streamlit styling nudged toward "friendly fintech" with a little injected CSS.
