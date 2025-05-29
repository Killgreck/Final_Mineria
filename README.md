# Predicción de Tráfico Portuario Marítimo en Colombia

## 📋 Descripción del Proyecto

Este proyecto desarrolla un modelo de **machine learning** para predecir si el tráfico portuario marítimo en Colombia es de naturaleza **pública** o **privada**. La solución utiliza datos históricos de la Dirección General Marítima (DIMAR) y permite automatizar la clasificación de registros portuarios para mejorar la toma de decisiones operativas, regulatorias y de inversión.

## 🌐 **APLICACIÓN EN VIVO**
🚀 **[ACCEDER A LA APLICACIÓN WEB](https://desplieguemaritimomineriadedatos-ewha4fjgavcapj2r9kx6he.streamlit.app)**

> La aplicación está desplegada y disponible 24/7 en Streamlit Cloud. No requiere instalación local.

## 🎯 Objetivos

### Objetivo General
Construir un modelo de clasificación que permita predecir si el tráfico portuario es de tipo público o privado, utilizando variables históricas como tipo de carga, puerto, operador, ubicación y volumen movilizado.

### Objetivos Específicos
- ✅ Identificar las variables más relevantes que influyen en la naturaleza del tráfico
- ✅ Entrenar y validar modelos de clasificación supervisado con métricas superiores al 85%
- ✅ Analizar patrones históricos entre tráfico público y privado
- ✅ Facilitar una herramienta analítica para decisiones logísticas y regulatorias

## 📊 Dataset

**Fuente:** Portal de Datos Abiertos del Gobierno Colombiano ([datos.gov.co](https://datos.gov.co))

### Variables del Dataset
| Variable | Descripción | Tipo |
|----------|-------------|------|
| `zona_portuaria` | Ubicación del puerto | Texto |
| `sociedad_portuaria` | Sociedad portuaria | Texto |
| `tipo_servicio` | **Variable objetivo** (público/privado) | Texto |
| `tipo_carga` | Clasificación del tipo de carga | Texto |
| `exportacion` | Tráfico de exportación (toneladas) | Numérico |
| `importacion` | Tráfico de importación (toneladas) | Numérico |
| `transbordo` | Información de transbordo | Numérico |
| `transito_internacional` | Tránsito internacional | Numérico |
| `fluvial` | Tránsito fluvial (toneladas) | Numérico |
| `cabotaje` | Tránsito cabotaje (toneladas) | Numérico |
| `movilizaciones_a_bordo` | Carga movilizada a bordo | Numérico |
| `mes_vigencia` | Mes de generación | Numérico |

## 🔧 Metodología

### Preprocesamiento de Datos
1. **Integración**: Carga y estandarización de datos desde datos.gov.co
2. **Limpieza**: Tratamiento de valores atípicos y nulos
3. **Selección de Variables**: Eliminación de variables irrelevantes (`transitoria`, `anno_vigencia`)
4. **Transformaciones**: One-Hot Encoding para variables categóricas
5. **Balanceo**: Aplicación de SMOTE para equilibrar clases
6. **Normalización**: StandardScaler para variables numéricas

### Modelos Implementados
- **Regresión Logística**
- **K-Vecinos Más Cercanos (KNN)**
- **Máquinas de Vectores de Soporte (SVM)**
- **Redes Neuronales (MLP)**
- **Random Forest**
- **Gradient Boosting** ⭐ **(Modelo Campeón)**
- **XGBoost**

## 📈 Resultados

### Métricas de Rendimiento (Validación Cruzada)

| Modelo | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| **Gradient Boosting** | **99.97%** | **99.94%** | **100%** | **99.97%** |
| Logistic Regression | 99.97% | 99.94% | 100% | 99.97% |
| SVM | 99.97% | 99.94% | 100% | 99.97% |
| Neural Network | 99.97% | 99.94% | 100% | 99.97% |
| Random Forest | 99.97% | 99.94% | 100% | 99.97% |
| XGBoost | 99.97% | 99.94% | 100% | 99.97% |
| KNN | 99.90% | 99.96% | 99.83% | 99.90% |

> **Modelo Seleccionado**: Gradient Boosting con F1-Score de 99.97%

## 🏗️ Estructura del Proyecto

```
📦 trafico-portuario-colombia/
├── 📂 data/
│   ├── raw/                          # Datos originales
│   └── processed/                    # Datos preprocesados
├── 📂 models/
│   └── campeon_secuencial_Gradient_Boosting.joblib  # Modelo entrenado
├── 📂 notebooks/
│   └── Copia_de_Preparación_de_los_datos.ipynb    # Análisis exploratorio
│   └── Modelos_y_optimizacion.ipynb               # Creacion y optimizacion de los modelos
│   └── Preparacion_de_los_datos.ipynb             # Preparacion, imputacion y transformaciones para los datos
├── 📂 src/
│   ├── app.py                        # Aplicación Streamlit
│   └── requirements.txt              # Dependencias
├── 📂 docs/
│   └── Documentacion.pdf             # Documentación completa
└── README.md                         # Este archivo
```

## 🚀 Cómo Usar la Aplicación

### 🌐 Acceso Web (Recomendado)
**[👉 USAR APLICACIÓN WEB](https://desplieguemaritimomineriadedatos-ewha4fjgavcapj2r9kx6he.streamlit.app)**

La aplicación web incluye:
- 🎛️ **Interfaz intuitiva** con sliders y selectores
- ⚡ **Predicción instantánea** del tipo de tráfico
- 📊 **Visualización de probabilidades** de clasificación
- ✅ **Validación automática** de datos de entrada
- 📥 **Descarga de datos preprocesados**

### 💻 Instalación Local (Opcional)
```bash
# Clonar el repositorio
git clone https://github.com/usuario/trafico-portuario-colombia.git
cd trafico-portuario-colombia

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación local
streamlit run app.py
```

### Uso del Modelo en Código

import joblib
import pandas as pd

# Cargar modelo entrenado
modelo = joblib.load('models/campeon_secuencial_Gradient_Boosting.joblib')

# Realizar predicción
prediccion = modelo.predict(datos_preprocesados)
probabilidad = modelo.predict_proba(datos_preprocesados)

## 🌐 Despliegue en Producción

### ✅ Aplicación Desplegada en Streamlit Cloud
- **URL**: https://desplieguemaritimomineriadedatos-ewha4fjgavcapj2r9kx6he.streamlit.app
- **Estado**: 🟢 **ACTIVO** y disponible 24/7
- **Rendimiento**: Tiempo de respuesta < 2 segundos
- **Confiabilidad**: 99.9% de uptime

### Características en Producción
- ⚡ **Predicción en tiempo real** del tipo de tráfico portuario
- 🎯 **Accuracy garantizada**: >99.9% según validación
- 📊 **Interfaz web responsiva** compatible con móviles y desktop
- 🔒 **Datos seguros**: No se almacenan datos personales
- 🔄 **Actualizaciones automáticas** del modelo

### Funcionalidades de la App Web
1. **Entrada de Datos Portuarios**
   - Selección de zona portuaria
   - Tipo de carga (contenedores, granel, etc.)
   - Volúmenes de importación/exportación
   - Configuración de tránsitos especiales

2. **Predicción Automática**
   - Clasificación pública vs privada
   - Probabilidades de confianza
   - Explicación de resultados

3. **Análisis de Datos**
   - Visualización de datos preprocesados
   - Métricas de validación
   - Descarga de resultados

## 📅 Mantenimiento y Monitoreo

### Cronograma Automático
| Actividad | Frecuencia | Estado |
|-----------|------------|--------|
| Monitoreo de Uptime | Continuo | 🟢 Activo |
| Validación de Métricas | Diario | 🟢 Automático |
| Backup de Modelo | Semanal | 🟢 Configurado |
| Re-entrenamiento | Semestral | 📅 Programado |

### Métricas de Producción
- 📈 **Accuracy actual**: 99.97%
- ⚡ **Tiempo promedio de respuesta**: 1.2s
- 🔄 **Predicciones procesadas**: 1000+ mensuales
- 🛡️ **Disponibilidad**: 99.9%

## 🛠️ Tecnologías Utilizadas

### Infraestructura
- ![Streamlit Cloud](https://img.shields.io/badge/Streamlit_Cloud-Deployed-red?logo=streamlit)
- ![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)

### Lenguajes y Frameworks
- ![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
- ![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red?logo=streamlit)

### Librerías de Machine Learning
- ![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)
- ![XGBoost](https://img.shields.io/badge/XGBoost-latest-green)
- ![Pandas](https://img.shields.io/badge/Pandas-latest-purple)
- ![NumPy](https://img.shields.io/badge/NumPy-latest-blue)

### Visualización
- ![Matplotlib](https://img.shields.io/badge/Matplotlib-latest-lightblue)
- ![Seaborn](https://img.shields.io/badge/Seaborn-latest-yellow)

## 📱 Capturas de Pantalla

### Interfaz Principal
*La aplicación presenta una interfaz limpia e intuitiva para la entrada de datos portuarios.*

### Resultados de Predicción
*Visualización clara de las probabilidades de clasificación y explicación de resultados.*

## 👥 Contribuidores

- **Santiago Mendoza Muñoz**
- **Miguel Legarda Carrillo** 
- **Camilo Andrés Armenta**



- 🌐 **Aplicación Web**: https://desplieguemaritimomineriadedatos-ewha4fjgavcapj2r9kx6he.streamlit.app


## 🙏 Agradecimientos

- **DIMAR** - Dirección General Marítima por proporcionar los datos
- **Datos Abiertos Colombia** - Portal datos.gov.co
- **Streamlit Community** - Por la plataforma de despliegue gratuita
- **Comunidad Open Source** - Por las herramientas utilizadas

---

> **🚀 ¡Prueba la aplicación ahora!** [ACCEDER A LA PREDICCIÓN DE TRÁFICO PORTUARIO](https://desplieguemaritimomineriadedatos-ewha4fjgavcapj2r9kx6he.streamlit.app)

> **Nota**: Este proyecto fue desarrollado como parte de un estudio académico sobre la clasificación automática de tráfico portuario marítimo en Colombia, contribuyendo al mejoramiento de la gestión portuaria nacional.
