# SpectraTrade — Clasificación de régimen de mercado vía representaciones tiempo-frecuencia

Predicción de señales de *trading* (**BUY / HOLD / SELL**) convirtiendo series
de precios en imágenes tiempo-frecuencia (espectrograma **STFT** y escalograma
**Wavelet** de Morlet) y clasificándolas con redes convolucionales (CNN propia
y ResNet-18 con *transfer learning*).

Trabajo final de **Visión por Computadora II** — Especialización en Inteligencia
Artificial (FIUBA / CEIA).

---

## 🎯 Resultados principales

### Experimento final — clasificación binaria (BUY vs SELL, STFT, horizonte 30d)

| Modelo | Test Acc | **F1-macro** | Params |
|--------|:--------:|:------------:|-------:|
| **CNN custom (v1)** | **61.30%** | **0.613** 🏆 | 1.45M |
| ResNet-18 | 53.25% | 0.532 | 11.2M |
| CNN v2 | 52.40% | 0.520 | 1.19M |
| EfficientNet-B0 | 50.34% | 0.478 | 4.17M |
| *Azar (2 clases)* | 50.0% | 0.500 | — |

### Experimento previo — clasificación ternaria (BUY / HOLD / SELL, STFT + Wavelet)

| Modelo | Repr. | Test Acc | Bal-Acc | **F1-macro** | Params |
|--------|-------|:--------:|:-------:|:------------:|-------:|
| **ResNet-18** | **STFT** | 62.3% | 37.4% | **0.369** 🏆 | 11.2M |
| ResNet-18 | Wavelet | 51.2% | 30.4% | 0.305 | 11.2M |
| CNN propia | STFT | 25.2% | 36.4% | 0.243 | 1.45M |
| CNN propia | Wavelet | 16.6% | 38.0% | 0.167 | 1.45M |
| *Azar (3 clases)* | — | 33.3% | 33.3% | 0.333 | — |

Métrica principal: **F1-macro** (no accuracy cruda, por desbalance de clases).
Gráficos y tablas en [`resultados/`](resultados/).

---

## 📁 Estructura del repositorio

```
.
├── notebooks/
│   └── proyecto_cv2_final.ipynb   # Notebook principal (recomendado)
├── src/
│   ├── generate_dataset.py        # Genera datasets STFT y Wavelet
│   ├── models.py                  # Definiciones de arquitecturas
│   ├── train_model.py             # Script de entrenamiento (CLI)
│   └── predict.py                 # Inferencia en tiempo real (CLI)
├── ui/
│   └── app.py                     # Demo interactiva Streamlit
├── models/                        # Pesos entrenados (.pth)
├── dataset_stft/                  # Dataset STFT (no versionado, regenerar)
├── dataset_wavelet/               # Dataset Wavelet (no versionado, regenerar)
├── resultados/                    # Tablas .csv con métricas
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 🚀 Configuración del entorno

### Opción recomendada: `uv`

```bash
# Instalar uv (si no está instalado)
pip install uv

# Crear entorno e instalar dependencias
uv sync

# Activar entorno
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### Alternativa: `venv` + `pip`

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

> Para entrenar con GPU: instalar PyTorch con soporte CUDA según tu versión en
> [pytorch.org](https://pytorch.org/get-started/locally/).

---

## 📓 Notebook principal (recomendado)

El flujo completo —generación de datos, entrenamiento de los 4 modelos,
comparación, barrido de horizontes y Grad-CAM— está en
[notebooks/proyecto_cv2_final.ipynb](notebooks/proyecto_cv2_final.ipynb).

**Ejecutar localmente:**

```bash
uv run jupyter lab
# o con el entorno activado:
jupyter lab
```

Abrir `notebooks/proyecto_cv2_final.ipynb` y ejecutar las celdas en orden.

**Ejecutar en Google Colab (recomendado para GPU):**

1. Subir `notebooks/proyecto_cv2_final.ipynb` a Colab.
2. `Runtime → Change runtime type → GPU`.
3. `Runtime → Run all`.

La celda de setup no regenera el dataset si ya existe. Los pesos se guardan en
`models/` y los resultados en `resultados/`.

> El notebook incluye los outputs de la corrida final para inspección sin re-ejecutar.

---

## 🖥️ Demo interactiva (Streamlit)

```bash
# Con el entorno activado:
python -m streamlit run ui/app.py
# o con uv:
uv run streamlit run ui/app.py
```

Abre `http://localhost:8501`. Permite:
- Elegir mercado y ticker (datos en vivo via yfinance).
- Cambiar entre **ResNet-STFT** y **ResNet-Wavelet**.
- Ver precio, espectrograma/escalograma y la señal **BUY/HOLD/SELL** con probabilidades.

> Requiere los `.pth` en `models/`. Si faltan, la UI cae a un *stub* y lo indica
> en la barra lateral.

---

## 🛠️ Scripts CLI (alternativa al notebook)

Los scripts en `src/` replican el pipeline del notebook y aceptan argumentos
por línea de comandos. Usar si se prefiere automatización o integración en pipelines.

### 1. Generar dataset

```bash
python src/generate_dataset.py --dataset both --verify
# Opciones: --dataset {stft|wavelet|both}, --start-date 2021-01-01, --window 90
```

### 2. Entrenar modelos

```bash
# Un modelo
python src/train_model.py --model resnet --epochs 20

# Todos los modelos
python src/train_model.py --model all --epochs 20

# Con dataset Wavelet y horizonte personalizado
python src/train_model.py --model cnn_v1 --dataset ./dataset_wavelet --horizon 7 --epochs 30

# Ver todas las opciones
python src/train_model.py --help
```

**Argumentos principales:**

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--model` | `resnet` | `resnet`, `efficientnet`, `cnn_v1`, `cnn_v2`, `all` |
| `--dataset` | `./dataset_stft` | Directorio del dataset |
| `--horizon` | `15` | Días a futuro (`3`, `7`, `15`, `30`) |
| `--epochs` | `20` | Épocas de entrenamiento |
| `--batch-size` | `256` | Tamaño de batch |
| `--output` | `./models` | Directorio de salida |

### 3. Inferencia en tiempo real

```bash
# Bitcoin con ResNet
python src/predict.py --ticker BTC-USD --model resnet --weights ./models/resnet_best.pth

# Apple con horizonte corto
python src/predict.py --ticker AAPL --model cnn_v1 --weights ./models/cnn_v1_best.pth --horizon 7

# Oro con Wavelet
python src/predict.py --ticker GC=F --model efficientnet --weights ./models/efficientnet_best.pth \
  --dataset-type wavelet

# Ver todas las opciones
python src/predict.py --help
```

### Modelos disponibles

| Model | Tipo | Params | Velocidad |
|-------|------|--------|-----------|
| `cnn_v1` | Custom | 1.45M | Rápido |
| `cnn_v2` | Custom | 370K | Muy rápido |
| `resnet` | Transfer learning | 11.2M | Medio |
| `efficientnet` | Transfer learning | ~5M | Medio |

---

## 📊 Datasets y modelos entrenados

- **Modelos (.pth) y datasets:** no versionados en el repositorio por tamaño.
  Disponibles en [Google Drive](https://drive.google.com/drive/u/0/folders/1rGp7yx1szeQyzjXx6y3OUwjwf3MLfiOX).
  Descargar y colocar los `.pth` en `models/` y los datasets en `dataset_stft/` / `dataset_wavelet/`.
- **Alternativamente**, los datasets se pueden regenerar con `src/generate_dataset.py`
  o la celda correspondiente del notebook.
- **Resultados:** [`resultados/`](resultados/) — `tabla_comparativa.csv`,
  `barrido_horizontes.csv`.

---

## 👥 Autores

- Sofía Belén Caselli
- Gaspar Rivollier
- Miguel Ángel Leiva Martínez
