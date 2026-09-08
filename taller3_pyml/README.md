# 🩺 Sistema de Diagnóstico Clínico con Machine Learning

Proyecto que entrena un **Random Forest** para clasificar 5 enfermedades (infarto, neumonía, gripe, ansiedad y gastroenteritis) a partir de 34 variables clínicas (signos vitales, factores de riesgo y síntomas), y expone el modelo en una app web con **Streamlit**.

## 📁 Estructura

```
├── Carga_datos/                  # Ejemplos de carga de datos (CSV, Excel, API, Web scraping)
├── Modelos_ML/RandomForest/
│   ├── 1.Crear_dataset.py        # Genera el dataset sintético (5.000 pacientes)
│   ├── 2.Entrenar_modelo.py      # Entrena el Random Forest y guarda el .pkl
│   └── 3.Predecir_enfermedad.py  # App Streamlit de diagnóstico
├── data/
│   └── dataset_medico_ampliado.csv   # Dataset generado (34 features + diagnóstico)
├── models/
│   └── modelo_random_forest_ampliado.pkl  # Modelo entrenado
└── requirements.txt
```

## 🚀 Instalación

```bash
pip install -r requirements.txt
```

## 🔄 Flujo de trabajo

### 1. Generar el dataset

```bash
python Modelos_ML/RandomForest/1.Crear_dataset.py
```

Genera `data/dataset_medico_ampliado.csv` con 5.000 registros sintéticos (34 columnas de entrada + `diagnostico`).

### 2. Entrenar el modelo

```bash
python Modelos_ML/RandomForest/2.Entrenar_modelo.py
```

Divide train/test (70/30), entrena un `RandomForestClassifier` (200 árboles) y guarda `models/modelo_random_forest_ampliado.pkl`. Muestra exactitud, reporte de clasificación, matriz de confusión e importancia de variables.

### 3. Lanzar la app de diagnóstico

```bash
streamlit run Modelos_ML/RandomForest/3.Predecir_enfermedad.py
```

Abre una interfaz con pestañas (signos vitales, factores de riesgo, síntomas) y devuelve el diagnóstico con probabilidades, nivel de urgencia y recomendación clínica.

> 💡 Dentro de la app hay una **tabla de ayuda** con los síntomas típicos de cada enfermedad para probar el modelo.

## ⚠️ Aviso

Herramienta educativa de apoyo al diagnóstico. **No sustituye** la evaluación de un profesional médico.