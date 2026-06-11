# SpectraTrade — Clasificación de régimen de mercado vía representaciones tiempo-frecuencia

Predicción de señales de *trading* (**BUY / HOLD / SELL**) convirtiendo series
de precios en imágenes tiempo-frecuencia (espectrograma **STFT** y escalograma
**Wavelet** de Morlet) y clasificándolas con redes convolucionales (CNN propia
y ResNet-18 con *transfer learning*).

Trabajo final de **Visión por Computadora II** — Especialización en Inteligencia
Artificial (FIUBA / CEIA).

---

## 🎯 Resultados principales

Comparación de 4 modelos bajo condiciones idénticas (test = año 2026, fuera de
muestra). Por el fuerte desbalance de clases, la métrica principal es
**F1-macro** / **balanced-accuracy**, no la *accuracy* cruda.

| Modelo | Repr. | Test Acc | Bal-Acc | **F1-macro** | Params |
|--------|-------|:--------:|:-------:|:------------:|-------:|
| **ResNet-18** | **STFT** | 62.3% | 37.4% | **0.369** 🏆 | 11.2M |
| ResNet-18 | Wavelet | 51.2% | 30.4% | 0.305 | 11.2M |
| CNN propia | STFT | 25.2% | 36.4% | 0.243 | 1.45M |
| CNN propia | Wavelet | 16.6% | 38.0% | 0.167 | 1.45M |
| *Azar (3 clases)* | — | 33.3% | 33.3% | 0.333 | — |

**Conclusiones clave:**
1. El *transfer learning* (ResNet-18) supera ampliamente a la CNN entrenada desde cero.
2. **STFT > Wavelet** en ambas arquitecturas (ablation central del proyecto).
3. El horizonte corto (7 días) maximiza el desempeño.
4. La señal direccional existe pero es **débil**: apenas por encima del azar.

Gráficos y tablas en [`resultados/`](resultados/). Detalle completo en el
[paper](paper/main.pdf).

---

## 📁 Estructura del repositorio

```
.
├── proyecto_final_resultados.ipynb   # ⭐ Notebook principal (con outputs ejecutados)
├── src/
│   ├── generate_dataset.py           # Genera datasets STFT y Wavelet
│   └── train_model.py                # Script de entrenamiento (versión standalone)
├── ui/
│   └── app.py                        # Demo Streamlit (modelo real conectado)
├── models/                           # Pesos entrenados (.pth) — los 4 modelos
├── resultados/                       # Tablas (.csv) y gráficos (.png)
├── paper/
│   ├── main.tex                      # Paper IEEE (LaTeX)
│   └── figs/                         # Figuras del paper
├── requirements.txt
└── README.md
```

> **Nota:** los datasets de imágenes (`dataset_stft/`, `dataset_wavelet/`) no se
> versionan por su tamaño (ver `.gitignore`). Se regeneran con el notebook (ver abajo).

---

## 🚀 Reproducir el proyecto

### 1. Entorno

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

pip install -r requirements.txt
```

Para entrenar / inferir con los modelos hace falta también PyTorch:

```bash
# CPU (suficiente para la UI / inferencia):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# o con GPU según tu CUDA (ver https://pytorch.org)
```

### 2. Generar el dataset

```bash
python src/generate_dataset.py --dataset both
# genera ./dataset_stft/ y ./dataset_wavelet/ con images/ + labels.csv
```

Opciones útiles: `--classes crypto stocks`, `--start 2020-01-01`,
`--verify` para chequear integridad.

### 3. Entrenar y comparar modelos

El flujo completo (generación → 4 modelos → comparación → barrido de
horizontes → Grad-CAM) está en el **notebook principal**, pensado para
ejecutarse en **Google Colab con GPU**:

1. Subir `proyecto_final_resultados.ipynb` a Colab.
2. `Runtime → Change runtime type → GPU`.
3. `Runtime → Run all`.

La celda de setup monta Google Drive (con *fallback* a `/content` si falla) y
**no regenera el dataset si ya existe**. Los pesos se guardan en
`models/model_<ARCH>-<REPR>.pth` y las tablas/gráficos en `resultados/`.

> El notebook ya incluye los outputs de la corrida final (métricas, tablas,
> gráficos) para inspección sin re-ejecutar.

### 4. Demo interactiva (Streamlit)

```bash
python -m streamlit run ui/app.py
```

Abre `http://localhost:8501`. Permite:
- Elegir mercado y ticker (yfinance en vivo).
- **Cambiar entre ResNet-STFT y ResNet-Wavelet** (cada uno genera su propia
  representación tiempo-frecuencia, idéntica a la de entrenamiento).
- Ver precio, espectrograma/escalograma y la señal **BUY/HOLD/SELL** con
  probabilidades por clase.

Requiere los `.pth` en `models/` (descargar del
[link a Drive](TODO_PEGAR_LINK_AQUI)). Si faltan, la UI cae a un *stub* y lo
avisa en la barra lateral.

---

## 🔬 Metodología (resumen)

| Etapa | Detalle |
|-------|---------|
| **Datos** | Precios diarios (yfinance), multi-activo: crypto, acciones, commodities, FX. Desde 2022. |
| **Preprocesamiento** | Log-returns → z-score, ventana deslizante de 60 días. |
| **STFT** | Hann 32, overlap 28, nfft 64. |
| **Wavelet** | CWT Morlet `cmor1.5-1.0`, escalas 2–30 días. |
| **Imágenes** | 64×64, colormap *magma*, frecuencias bajas abajo. |
| **Etiquetado** | BUY/HOLD/SELL con umbral adaptativo `k·ATR·√h` (k=0.7). |
| **Split** | Cronológico: train/val 2022–2025, **test = 2026** (fuera de muestra). |
| **Desbalance** | WeightedRandomSampler + CrossEntropy ponderada + label smoothing. |
| **Métricas** | Balanced-accuracy y F1-macro (no accuracy cruda). |

---

## 📊 Datasets y modelos

- **Modelos entrenados (.pth):** ⚠️ no versionados por su tamaño (~100 MB).
  Descargar desde Drive y colocar en `models/`:
  **[📥 Link a modelos en Drive](TODO_PEGAR_LINK_AQUI)**
  (4 archivos: `model_RESNET-STFT.pth`, `model_RESNET-Wavelet.pth`,
  `model_CNN-STFT.pth`, `model_CNN-Wavelet.pth`).
- **Resultados:** [`resultados/`](resultados/) — `tabla_comparativa.csv`,
  `barrido_horizontes.csv`, `comparacion_modelos.png`, `barrido_horizontes.png`.
- **Datasets de imágenes:** se regeneran con `src/generate_dataset.py` (no versionados).

---

## 📄 Paper

El informe en formato IEEE Conference está en [`paper/main.tex`](paper/main.tex).
Compilar con:

```bash
cd paper
pdflatex main.tex && pdflatex main.tex   # 2 pasadas (referencias cruzadas)
```

---

## 👥 Autores

- Sofía Belén Caselli
- Gaspar Rivollier
- Miguel Ángel Leiva Martínez
