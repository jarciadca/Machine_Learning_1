\# Regresión Lineal Simple — Boston Housing Dataset



!\[Python](https://img.shields.io/badge/Python-3.10+-blue.svg)

!\[scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-orange.svg)

!\[License](https://img.shields.io/badge/License-Educational-green.svg)



Ejercicio de Machine Learning que implementa un modelo de \*\*regresión lineal simple\*\* para predecir el precio medio de viviendas en Boston a partir del número de habitaciones.



\---



\## 📋 Descripción



Este proyecto aplica los conceptos fundamentales de \*\*aprendizaje supervisado\*\* usando `scikit-learn`:



\- Carga y preprocesamiento del dataset Boston Housing

\- División de datos en entrenamiento (80%) y prueba (20%)

\- Entrenamiento de un modelo de regresión lineal simple

\- Evaluación con métricas (MSE, RMSE, R²)

\- Visualización gráfica del ajuste

\- Diagnóstico de overfitting comparando R² de train y test



El objetivo es \*\*predecir `MEDV`\*\* (precio medio de la vivienda en miles de USD) a partir de \*\*`RM`\*\* (número promedio de habitaciones).



\---



\## 🎯 Objetivos de aprendizaje



\- Comprender el flujo de un proyecto de ML supervisado

\- Aplicar `train\_test\_split` para evitar overfitting

\- Entrenar un modelo con `LinearRegression`

\- Interpretar métricas: \*\*MSE\*\*, \*\*RMSE\*\*, \*\*R²\*\*

\- Diagnosticar overfitting comparando R² de train y test

\- Visualizar el ajuste con `matplotlib`



\---



\## 📁 Estructura del proyecto



```

1-REGRESION\_LINEAR\_SIMPLE/

│

├── regresion\_linear\_simple.ipynb   # Notebook principal con el ejercicio

├── requirements.txt                # Dependencias del proyecto

├── README.md                       # Este archivo

├── .gitignore                      # Archivos excluidos de Git

└── .venv/                          # Entorno virtual (no se sube a Git)

```



\---



\## 🛠️ Requisitos



\- Python 3.10+

\- pip



\### Dependencias



```

numpy

pandas

matplotlib

scikit-learn

jupyter

```



\---



\## 🚀 Instalación y uso



\### 1. Clonar el repositorio



```bash

git clone https://github.com/tu-usuario/regresion-linear-simple.git

cd regresion-linear-simple

```



\### 2. Crear y activar el entorno virtual



\*\*Windows:\*\*

```bash

python -m venv .venv

.venv\\Scripts\\activate

```



\*\*Linux / macOS:\*\*

```bash

python3 -m venv .venv

source .venv/bin/activate

```



\### 3. Instalar dependencias



```bash

python -m pip install --upgrade pip

python -m pip install -r requirements.txt

```



\### 4. Abrir el notebook



Abre `regresion\_linear\_simple.ipynb` en VS Code o Jupyter:



```bash

jupyter notebook

```



\---



\## 📊 Dataset



\*\*Boston Housing Dataset\*\* — 506 muestras, 13 features + 1 target.



⚠️ \*\*Nota ética:\*\* este dataset fue removido de `scikit-learn` en la versión 1.2 por problemas éticos en su construcción (la variable `B` asume que la autosegregación racial impacta positivamente en el precio). Se usa aquí \*\*solo con fines educativos\*\*.



\### Variables principales



| Variable | Descripción |

|---|---|

| `RM` | Número promedio de habitaciones por vivienda |

| `MEDV` | Valor mediano de viviendas (target, en miles de USD) |

| `CRIM` | Tasa de criminalidad per cápita |

| `LSTAT` | % de población de bajo estatus socioeconómico |

| `PTRATIO` | Ratio alumno-profesor |

| `TAX` | Tasa de impuesto a la propiedad |

| `AGE` | Proporción de unidades construidas antes de 1940 |

| `DIS` | Distancia a centros de empleo |

| `RAD` | Índice de accesibilidad a autopistas |

| `NOX` | Concentración de óxidos nítricos |

| `INDUS` | Proporción de acres de negocios no minoristas |

| `ZN` | Proporción de terreno residencial para lotes grandes |

| `CHAS` | Variable dummy del río Charles |

| `B` | ⚠️ Variable problemática (ver nota ética) |



\---



\## 🔬 Metodología



\### 1. Carga de datos



Se descarga el dataset original desde el repositorio de Carnegie Mellon y se reconstruye como `DataFrame` de pandas:



```python

import pandas as pd

import numpy as np



data\_url = "http://lib.stat.cmu.edu/datasets/boston"

raw\_df = pd.read\_csv(data\_url, sep=r"\\s+", skiprows=22, header=None)

data = np.hstack(\[raw\_df.values\[::2, :], raw\_df.values\[1::2, :2]])

target = raw\_df.values\[1::2, 2]



column\_names = \['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',

&#x20;               'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

df = pd.DataFrame(data, columns=column\_names)

df\['MEDV'] = target

```



\### 2. División train/test



```python

from sklearn.model\_selection import train\_test\_split



x\_train, x\_test, y\_train, y\_test = train\_test\_split(

&#x20;   x, y, test\_size=0.2, random\_state=42

)

```



\- 80% entrenamiento (405 casas)

\- 20% prueba (101 casas)

\- `random\_state=42` para reproducibilidad



\### 3. Entrenamiento del modelo



```python

from sklearn.linear\_model import LinearRegression



lr = LinearRegression()

lr.fit(x\_train, y\_train)

```



Busca la recta `y = m·x + b` que minimiza el error cuadrático medio (OLS).



\### 4. Predicciones



```python

y\_pred\_train = lr.predict(x\_train)

y\_pred\_test  = lr.predict(x\_test)

```



\### 5. Métricas



```python

from sklearn.metrics import mean\_squared\_error, r2\_score



r2\_train = r2\_score(y\_train, y\_pred\_train)

r2\_test  = r2\_score(y\_test,  y\_pred\_test)

mse\_test = mean\_squared\_error(y\_test, y\_pred\_test)

rmse\_test = mse\_test \*\* 0.5

```



| Métrica | Fórmula | Interpretación |

|---|---|---|

| \*\*MSE\*\* | (1/n)Σ(y\_real - y\_pred)² | Error promedio al cuadrado |

| \*\*RMSE\*\* | √MSE | Error en unidades originales |

| \*\*R²\*\* | 1 - (SS\_res / SS\_tot) | % de varianza explicada |



\---



\## 📈 Resultados



\### Métricas del modelo



| Métrica | Valor |

|---|---|

| Pendiente (coef) | \~8.89 |

| Intercepto | \~-33.39 |

| R² Train | \~0.49 |

| R² Test | \~0.46 |

| Diferencia R² | \~0.03 |

| MSE Test | \~52.45 |

| RMSE Test | \~7.24 |



\### Interpretación



\- \*\*Pendiente \~8.89:\*\* cada habitación adicional incrementa el precio en \~$8,890

\- \*\*R² Test \~0.46:\*\* el modelo explica el \~46% de la variabilidad de los precios

\- \*\*Diferencia R² \~0.03:\*\* no hay overfitting (el modelo generaliza bien)

\- \*\*RMSE \~7.24:\*\* error promedio de ±$7,240 en las predicciones



\### Ecuación resultante



```

MEDV = 8.89 · RM - 33.39

```



\*\*Ejemplo:\*\* una casa con 6 habitaciones:

```

MEDV = 8.89 · 6 - 33.39 = 19.95  →  \~$19,950

```



\---



\## 📊 Visualización



El notebook genera un gráfico de dispersión con:



\- \*\*Puntos azules\*\* → precios reales (conjunto de test)

\- \*\*Línea roja\*\* → recta de regresión ajustada



```python

import matplotlib.pyplot as plt



plt.figure(figsize=(8, 5))

plt.scatter(x\_test, y\_test, alpha=0.5, label='Datos reales')

plt.plot(x\_test, y\_pred\_test, color='red', linewidth=2, label='Regresión')

plt.xlabel('Nº habitaciones (RM)')

plt.ylabel('Precio medio (MEDV)')

plt.title('Regresión Lineal Simple — RM vs MEDV')

plt.legend()

plt.grid(alpha=0.3)

plt.show()

```



\---



\## 🧠 Diagnóstico del modelo



| Situación | Train R² | Test R² | Diferencia | Diagnóstico |

|---|---|---|---|---|

| Underfitting | 0.30 | 0.28 | 0.02 | Modelo muy simple |

| \*\*✅ Buen ajuste\*\* | \*\*0.49\*\* | \*\*0.46\*\* | \*\*0.03\*\* | \*\*Correcto\*\* |

| Overfitting | 0.99 | 0.40 | 0.59 | Memorizó |



\*\*Conclusión:\*\* el modelo está bien ajustado. La diferencia entre train y test es pequeña (\~3%), lo que indica que \*\*generaliza correctamente\*\* a datos nuevos.



\---



\## 🧩 Conceptos clave aplicados



| Concepto | Descripción |

|---|---|

| \*\*Train/Test Split\*\* | División 80/20 para evitar overfitting |

| \*\*Overfitting\*\* | El modelo memoriza en vez de aprender |

| \*\*Underfitting\*\* | El modelo es demasiado simple |

| \*\*MSE\*\* | Error cuadrático medio (más bajo = mejor) |

| \*\*RMSE\*\* | Raíz del MSE, en unidades originales |

| \*\*R²\*\* | % de varianza explicada (más cerca de 1 = mejor) |

| \*\*OLS\*\* | Mínimos cuadrados ordinarios (método de ajuste) |



\---



\## 🧠 Conclusiones



\- La regresión lineal simple con \*\*una sola variable (`RM`)\*\* logra un R² de \~0.46

\- El 54% restante de la varianza depende de \*\*otras variables\*\* no incluidas en este modelo

\- El modelo \*\*no presenta overfitting\*\* (diferencia train/test \~3%)

\- Para mejorar el rendimiento sería necesario pasar a \*\*regresión lineal múltiple\*\* con las 13 features



\---



\## 🚀 Próximos pasos



\- \[ ] Implementar regresión lineal múltiple (13 features)

\- \[ ] Analizar correlaciones entre variables

\- \[ ] Probar regularización (Ridge, Lasso)

\- \[ ] Validación cruzada (cross-validation)

\- \[ ] Análisis de residuos

\- \[ ] Migrar a un dataset sin problemas éticos (California Housing)



\---



\## 📚 Referencias



\- \[scikit-learn — LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear\_model.LinearRegression.html)

\- \[scikit-learn — train\_test\_split](https://scikit-learn.org/stable/modules/generated/sklearn.model\_selection.train\_test\_split.html)

\- \[Harrison, D. and Rubinfeld, D.L. (1978). Hedonic prices and the demand for clean air](https://www.researchgate.net/publication/4974606\_Hedonic\_housing\_prices\_and\_the\_demand\_for\_clean\_air)

\- \[M. Carlisle — Racist data destruction?](https://medium.com/@docintangible/racist-data-destruction-113e3eff54a8)



\---



\## 👤 Autor



\*\*Tu Nombre\*\*

\- GitHub: \[@tu-usuario](https://github.com/tu-usuario)

\- Email: tu@email.com



\---



\## 📄 Licencia



Este proyecto es de uso \*\*educativo\*\*. El dataset Boston tiene restricciones éticas conocidas — ver sección de referencias.



\---



\## 🏷️ Tags



`machine-learning` `regression` `linear-regression` `scikit-learn` `python` `data-science` `boston-housing` `jupyter-notebook`

