# Taller 3 – Despliegue de Modelos de Machine Learning con FastAPI

Este repositorio reúne **tres modelos de Machine Learning** entrenados con distintos algoritmos y desplegados en producción usando herramientas diferentes para cada caso. El objetivo del taller es demostrar el ciclo completo de un proyecto de ML: generación/carga de datos, entrenamiento, serialización del modelo, construcción de una API/interfaz y despliegue en la nube.

| Modelo | Algoritmo | Caso de uso | Backend | Frontend | Despliegue |
|---|---|---|---|---|---|
| 1 | **Random Forest** (Clasificación) | Diagnóstico clínico asistido | Streamlit (todo en uno) | Streamlit | **Streamlit Community Cloud** |
| 2 | **Regresión Lineal** | Predicción del precio de una vivienda según m² | FastAPI | Django | **Railway** (2 servicios independientes) |
| 3 | **Visión Artificial** (OpenCV / Haar Cascade) | Detección de rostros en imágenes/video | Flask (Python serverless) | HTML + JS + Bootstrap | **Vercel** |

---

## 📁 Estructura del repositorio

```
taller3_fastapi/
├── taller3_pyml/
│   ├── Carga_datos/                        # Notebooks de ingesta (CSV, Excel, API, Web scraping)
│   ├── data/ , models/                     # Copias de datos y modelo Random Forest en la raíz del proyecto
│   ├── Modelos_ML/
│   │   ├── RandomForest/                   # Modelo 1: Diagnóstico clínico
│   │   │   ├── 1.Crear_dataset.py
│   │   │   ├── 2.Entrenar_modelo.py
│   │   │   ├── 3.Predecir_enfermedad.py    # App Streamlit
│   │   │   ├── data/dataset_medico_ampliado.csv
│   │   │   └── models/modelo_random_forest_ampliado.pkl
│   │   │
│   │   ├── RegresionLineal/                # Modelo 2: Precio de vivienda
│   │   │   ├── backend/                    # API FastAPI
│   │   │   │   ├── main.py
│   │   │   │   ├── train.py
│   │   │   │   └── models/linear_model.joblib
│   │   │   └── frontend/                   # Cliente Django
│   │   │       ├── prediccion/ (views, urls, templates)
│   │   │       └── config/ (settings, urls)
│   │   │
│   │   └── Visionartificial/               # Modelo 3: Detección de rostros
│   │       ├── index.ipynb                 # Prototipo en notebook (OpenCV local)
│   │       └── py_img/                     # App web desplegada
│   │           ├── api/index.py            # Backend Flask
│   │           ├── public/                 # Frontend (HTML/CSS/JS)
│   │           ├── haarcascade_frontalface_default.xml
│   │           └── vercel.json
│   └── requirements.txt
└── requirements.txt
```

---

## 🩺 Modelo 1 — Random Forest: Diagnóstico Clínico

### Descripción
Sistema de apoyo al diagnóstico que clasifica al paciente en una de **5 posibles enfermedades** — `infarto`, `neumonía`, `gripe`, `ansiedad`, `gastroenteritis` — a partir de **34 variables clínicas** (signos vitales, factores de riesgo, síntomas cardiorrespiratorios, neuropsiquiátricos y gastrointestinales).

### ¿Cómo funciona el modelo?
Un **Random Forest** es un algoritmo de *ensemble learning* que entrena muchos árboles de decisión sobre subconjuntos aleatorios de datos y variables, y combina sus predicciones (votación por mayoría en clasificación). Esto reduce el sobreajuste que tendría un único árbol y suele ofrecer buena precisión y robustez ante variables ruidosas o correlacionadas, además de permitir calcular la **importancia de cada variable** en la decisión final.

### Pipeline de datos y entrenamiento
1. **`1.Crear_dataset.py`** — Genera un dataset sintético de 5.000 pacientes (`np.random.seed(42)`), simulando de forma realista los rangos fisiológicos y probabilidades de síntomas propios de cada enfermedad (ej. en infarto: FC 100–140, SatO₂ 88–96%, dolor opresivo irradiado; en neumonía: fiebre 38–39.5°C, tos productiva, SatO₂ 85–95%). Guarda el resultado en `data/dataset_medico_ampliado.csv` (34 columnas de entrada + `diagnostico`).
2. **`2.Entrenar_modelo.py`**:
   - Separa `X` (34 features clínicas) e `y` (`diagnostico`).
   - Split **70/30** (`train_test_split`, `stratify=y`, `random_state=42`).
   - Entrena `RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)`.
   - Evalúa con `accuracy`, `classification_report` y matriz de confusión, y calcula la importancia relativa (%) de cada variable.
   - Serializa el modelo con `joblib` en `models/modelo_random_forest_ampliado.pkl`.
3. **`3.Predecir_enfermedad.py`** — Aplicación **Streamlit** que carga el `.pkl` (con `@st.cache_resource`), presenta un formulario por pestañas (signos vitales, factores de riesgo, síntomas cardiorrespiratorios, otros síntomas), reordena las variables según `model.feature_names_in_` y al predecir muestra:
   - Diagnóstico principal y nivel de urgencia asociado.
   - Probabilidad/confianza de la predicción.
   - Gráfico de barras horizontal (Plotly) con la probabilidad de cada una de las 5 enfermedades.
   - Recomendación clínica textual según el diagnóstico.

### Stack técnico
`Python` · `pandas` / `numpy` · `scikit-learn` (`RandomForestClassifier`) · `joblib` · `Streamlit` · `Plotly`

### Despliegue
Desplegado en **Streamlit Community Cloud**, que ejecuta directamente el script `3.Predecir_enfermedad.py` sobre un contenedor gestionado por Streamlit (sin necesidad de configurar servidor, Dockerfile ni balanceador — el propio hosting instala `requirements.txt` y levanta la app).

🔗 **URL de producción:** https://taller3fastapi-idi9x4amfkzdwz7pykwywg.streamlit.app/

> ⚠️ Herramienta educativa de apoyo al diagnóstico. No sustituye la evaluación de un profesional médico.

---

## 🏠 Modelo 2 — Regresión Lineal: Precio de Vivienda según Superficie

### Descripción
Modelo que predice el **precio estimado de una vivienda (COP)** a partir de su **superficie en m²**, expuesto mediante una API REST (FastAPI) y consumido por una aplicación web (Django).

### ¿Cómo funciona el modelo?
Una **Regresión Lineal simple** ajusta la recta `precio = coeficiente × área + intercepto` que minimiza el error cuadrático entre los precios observados y los predichos. Es el modelo más simple de la familia de regresión y sirve como línea base interpretable: el coeficiente indica cuánto aumenta el precio por cada m² adicional.

### Entrenamiento (`backend/train.py`)
Se entrena con `sklearn.linear_model.LinearRegression` sobre un dataset de ejemplo (6 observaciones) que relaciona superficie (m²) y precio (COP):

```python
x = [[40], [50], [60], [85], [100], [120]]
y = [210000000, 300000000, 350000000, 500000000, 600000000, 700000000]
```

El modelo entrenado se serializa con `joblib` en `backend/models/linear_model.joblib`. En el `Dockerfile` del backend, este entrenamiento se ejecuta automáticamente en tiempo de build (`RUN python train.py`), por lo que el artefacto siempre se regenera al desplegar.

### Arquitectura: dos servicios independientes
Este modelo, a diferencia de los otros dos, se desplegó como **dos aplicaciones separadas que se comunican por HTTP**:

**1) Backend — API FastAPI (`backend/main.py`)**
- Framework: **FastAPI** + **Pydantic** para validación de esquemas.
- Carga el modelo `.joblib` al iniciar.
- Endpoints:
  - `GET /` → *health check*, retorna estado del servicio y si el modelo se cargó correctamente.
  - `POST /predict` → recibe `{"area_m2": float}` y retorna `{"area_m2": ..., "predicted_price": ...}`. Si el modelo no cargó, responde `503`.
- Expone documentación interactiva automática vía Swagger UI en `/docs`.
- Contenedorizado con Docker (`python:3.10-slim`), ejecutado con `uvicorn`.

**2) Frontend — Aplicación Django (`frontend/`)**
- Framework: **Django** (con `requests` para consumir la API y `gunicorn` como servidor de producción).
- La vista `predecir_precio` (`prediccion/views.py`) recibe el formulario, valida el área ingresada y hace un `POST` al backend usando la variable de entorno `FASTAPI_URL` (definida en `config/settings.py`), manejando errores de conexión, timeout y respuestas no exitosas.
- Renderiza el resultado en la plantilla `prediccion/index.html`.
- También contenedorizado con Docker, ejecutado con `gunicorn`.

### Stack técnico
`FastAPI` · `Pydantic` · `scikit-learn` (`LinearRegression`) · `joblib` · `uvicorn` — Backend
`Django` · `requests` · `gunicorn` — Frontend
`Docker` para ambos servicios

### Despliegue
Ambos servicios se desplegaron de forma independiente en **Railway**, cada uno desde su propio `Dockerfile`. El frontend Django apunta al backend mediante la variable de entorno `FASTAPI_URL`, configurada con la URL pública del backend en Railway.

🔗 **API / Documentación Swagger (backend):** https://backend-fastapi-production-df87.up.railway.app/docs
🔗 **Aplicación web (frontend):** https://frontedfastapi-production.up.railway.app/

---

## 📷 Modelo 3 — Visión Artificial: Detección de Rostros

### Descripción
Aplicación web de **detección de rostros en tiempo real**, tanto en imágenes subidas por el usuario como en el flujo de video de la webcam, que dibuja un recuadro sobre cada rostro detectado y muestra el conteo total.

### ¿Cómo funciona el modelo?
A diferencia de los dos modelos anteriores (que son modelos estadísticos "clásicos" entrenados sobre datos tabulares), este componente usa un **Clasificador en Cascada de Haar** (`Haar Cascade`, `haarcascade_frontalface_default.xml`) de OpenCV — un modelo *pre-entrenado* de visión artificial. Funciona evaluando en múltiples escalas pequeñas regiones rectangulares de la imagen (en escala de grises) contra patrones de contraste típicos de un rostro (ojos más oscuros que la frente, puente nasal más claro, etc.), descartando rápidamente regiones que no cumplen el patrón en cascada de filtros. No requirió entrenamiento propio: se reutiliza el modelo Haar ya entrenado por OpenCV y el proyecto se enfoca en integrarlo dentro de una API y una interfaz usable.

### Backend (`py_img/api/index.py`)
- Framework: **Flask**.
- Carga el clasificador `cv2.CascadeClassifier(haarcascade_frontalface_default.xml)`.
- Endpoint `POST /api/detect`:
  1. Recibe la imagen (`multipart/form-data`).
  2. La decodifica con OpenCV (`cv2.imdecode`) y la convierte a escala de grises.
  3. Ejecuta `detectMultiScale(scaleFactor=1.1, minNeighbors=5, minSize=(40,40))` para localizar los rostros.
  4. Dibuja un rectángulo verde sobre cada rostro detectado en una copia de la imagen original.
  5. Codifica la imagen resultante en Base64 (JPEG) y responde en JSON: `{ success, faces_detected, image }`.
- También sirve el frontend estático (`index.html`, `style.css`, `script.js`) desde la carpeta `public/`.

### Frontend (`py_img/public/`)
- HTML + Bootstrap + JavaScript "vanilla".
- Dos modos de uso, controlados por `script.js`:
  - **Subida de imagen**: arrastrar y soltar (*drag & drop*) o seleccionar archivo, que se envía al endpoint `/api/detect`.
  - **Cámara web**: captura frames del `<video>` sobre un `<canvas>`, los envía periódicamente al backend y muestra el resultado procesado con los rostros resaltados casi en tiempo real.
- Barra de carga y conteo de rostros detectados en pantalla.

### Stack técnico
`Python` · `Flask` · `OpenCV` (`opencv-python-headless`) · `NumPy` — Backend
`HTML` / `CSS` / `JavaScript` / `Bootstrap` — Frontend

### Despliegue
Desplegado en **Vercel** como función serverless de Python (`@vercel/python`), configurado mediante `vercel.json`: las rutas `/api/*` se enrutan al backend Flask (`api/index.py`) y el resto de rutas sirven los archivos estáticos de `public/` (`@vercel/static`).

🔗 **URL de producción:** https://taller3-fastapi-33876kakk-juan-ea46.vercel.app/

---

## 🧩 Comparativa técnica de despliegue

| Aspecto | Random Forest | Regresión Lineal | Visión Artificial |
|---|---|---|---|
| Tipo de modelo | Clasificación (ensemble) | Regresión | Modelo pre-entrenado (clásico de CV) |
| Librería de ML | scikit-learn | scikit-learn | OpenCV |
| Serialización | `joblib` (.pkl) | `joblib` (.joblib) | N/A (XML de Haar Cascade) |
| Backend | Streamlit (todo integrado) | FastAPI | Flask |
| Frontend | Streamlit (mismo proceso) | Django (servicio separado) | HTML/JS estático |
| Contenedor | No (hosting nativo de Streamlit) | Sí, Docker (2 imágenes) | No (función serverless) |
| Plataforma de despliegue | Streamlit Community Cloud | Railway | Vercel |
| Comunicación front-back | N/A (monolítico) | HTTP REST (`FASTAPI_URL`) | HTTP REST (`/api/detect`) |

---

## ⚙️ Ejecución local

### Random Forest
```bash
cd taller3_pyml
pip install -r requirements.txt
python Modelos_ML/RandomForest/1.Crear_dataset.py
python Modelos_ML/RandomForest/2.Entrenar_modelo.py
streamlit run Modelos_ML/RandomForest/3.Predecir_enfermedad.py
```

### Regresión Lineal
```bash
# Backend
cd taller3_pyml/Modelos_ML/RegresionLineal/backend
pip install -r requirements.txt
python train.py
uvicorn main:app --reload

# Frontend (en otra terminal)
cd taller3_pyml/Modelos_ML/RegresionLineal/frontend
pip install -r requirements.txt
export FASTAPI_URL=http://127.0.0.1:8000   # En Windows: set FASTAPI_URL=...
python manage.py runserver
```

### Visión Artificial
```bash
cd taller3_pyml/Modelos_ML/Visionartificial/py_img
pip install -r requirements.txt
cd api
python index.py   # http://localhost:5000
```

---

## ⚠️ Aviso general
Estos modelos fueron desarrollados con fines **educativos** como parte de un taller académico. El modelo de diagnóstico clínico no sustituye la evaluación de un profesional de la salud, y las estimaciones de precio de vivienda se basan en un conjunto de datos de ejemplo muy reducido, por lo que no deben usarse como referencia comercial real.
