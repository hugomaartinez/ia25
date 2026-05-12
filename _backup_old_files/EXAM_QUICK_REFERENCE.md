# 📋 Guía Rápida para Examen - IA

**Guardar como PDF e imprimir para llevar al examen (sin internet)**

---

## 1️⃣ WORKFLOW GENERAL - CUALQUIER PROBLEMA

```python
# PASO 1: Cargar datos
import pandas as pd
df = pd.read_csv('data.csv')

# PASO 2: Exploración (EDA)
df.head()              # Primeras filas
df.info()              # Tipos de datos
df.describe()          # Estadísticas
df.isnull().sum()      # Valores faltantes
df.duplicated().sum()  # Duplicados

# PASO 3: Limpieza
df = df.dropna()                    # Eliminar NaNs
df = df[df['col'] != 'valor']       # Filtrar
df['col'] = df['col'].fillna(0)     # Rellenar

# PASO 4: Preparación (Features + Target)
X = df[['feat1', 'feat2', ...]]  # Features
y = df['target']                   # Target

# PASO 5: Train/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# PASO 6: Normalización (SI ES NECESARIO)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PASO 7: Entrenar modelo
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# PASO 8: Predicciones
y_pred = model.predict(X_test)

# PASO 9: Evaluar
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
print(accuracy_score(y_test, y_pred))
```

---

## 2️⃣ SELECCIONAR MODELO CORRECTO

### 🟢 REGRESIÓN (Predecir número continuo)
```python
# Problema: Predecir precio, temperatura, ventas
# Target: números reales (10.5, 100.3, etc)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100)

# EVALUAR con:
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse}, R²: {r2}")
```

### 🔵 CLASIFICACIÓN (Predecir categoría)
```python
# Problema: ¿Es spam? ¿Qué animal? ¿Iris tipo A, B o C?
# Target: categorías (0, 1, 2, ...)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)

from sklearn.svm import SVC
model = SVC(kernel='rbf')

# EVALUAR con:
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)
```

### ⚫ CLUSTERING (Agrupar puntos - SIN etiquetas)
```python
# Problema: Agrupar clientes, documentos
# NO hay "target" - buscar patrones

from sklearn.cluster import KMeans
model = KMeans(n_clusters=3, random_state=42)
clusters = model.fit_predict(X)

# EVALUAR con:
from sklearn.metrics import silhouette_score
score = silhouette_score(X, clusters)
print(f"Silhouette Score: {score}")  # Más cerca de 1 = mejor
```

---

## 3️⃣ MÉTRICAS RÁPIDAS

### Para REGRESIÓN:
| Métrica | Fórmula | Mejor | Código |
|---------|---------|-------|--------|
| **MSE** | (Σ(y-ŷ)²)/n | ↓ menor | `mean_squared_error(y_test, y_pred)` |
| **RMSE** | √MSE | ↓ menor | `np.sqrt(mean_squared_error(...))` |
| **MAE** | Σ\|y-ŷ\|/n | ↓ menor | `mean_absolute_error(y_test, y_pred)` |
| **R²** | 1 - (SS_res/SS_tot) | ↑ más alto | `r2_score(y_test, y_pred)` |

### Para CLASIFICACIÓN:
| Métrica | Usa cuando |
|---------|-----------|
| **Accuracy** | Clases balanceadas |
| **Precision** | Importa evitar falsos positivos (spam detection) |
| **Recall** | Importa detectar todos (diagnósticos médicos) |
| **F1-Score** | Balance entre precision/recall |

```python
# Confusion Matrix:
#           Predicho 0    Predicho 1
# Real 0    TN            FP
# Real 1    FN            TP

# Precision = TP / (TP + FP)  → De mis predicciones positivas, cuántas son correctas
# Recall = TP / (TP + FN)      → De los positivos reales, cuántos detecté
# F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

---

## 4️⃣ PIPELINES (El workflow profesional)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Ventaja: Normalización se aplica automáticamente
```

---

## 5️⃣ VALIDACIÓN CRUZADA (K-Fold)

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    estimator=model,
    X=X_train,
    y=y_train,
    cv=5  # 5-fold
)
print(f"Scores: {scores}")
print(f"Media: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

---

## 6️⃣ MANEJO DE DATOS

### Datos faltantes (NaN):
```python
# Ver
df.isnull().sum()

# Eliminar filas
df = df.dropna()

# Rellenar
df['col'].fillna(df['col'].mean(), inplace=True)
df['col'].fillna(0, inplace=True)
```

### Categorías (texto):
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['col_encoded'] = le.fit_transform(df['col'])

# O One-Hot Encoding:
df = pd.get_dummies(df, columns=['col_categorica'])
```

### Normalización:
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler (media 0, std 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MinMaxScaler (rango 0-1)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)
```

---

## 7️⃣ VISUALIZACIÓN

```python
import matplotlib.pyplot as plt

# Gráfico de dispersión
plt.scatter(X, y)
plt.xlabel('Feature')
plt.ylabel('Target')
plt.show()

# Matriz de confusión
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
plt.show()

# Importancia de features
plt.barh(feature_names, model.feature_importances_)
plt.show()
```

---

## 8️⃣ HIPERPARÁMETROS COMUNES

### Logistic Regression:
```python
LogisticRegression(C=1.0, max_iter=100, random_state=42)
# C: Regularización inversa (↓ más regularización)
```

### Decision Tree:
```python
DecisionTreeClassifier(max_depth=5, min_samples_split=5, random_state=42)
# max_depth: Profundidad máxima
# min_samples_split: Muestras mínimas para dividir
```

### Random Forest:
```python
RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
# n_estimators: Número de árboles
# max_depth: Profundidad de cada árbol
```

### KMeans:
```python
KMeans(n_clusters=3, max_iter=300, random_state=42)
# n_clusters: Número de grupos
```

### SVM:
```python
SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
# kernel: 'linear', 'rbf', 'poly'
# C: Regularización
```

---

## 9️⃣ PANDAS ESSENTIALS

```python
# Seleccionar columnas
df[['col1', 'col2']]
df['col']

# Filtrar
df[df['col'] > 5]
df[df['col'] == 'valor']
df[(df['col1'] > 5) & (df['col2'] == 'X')]

# Agregar
df.groupby('col').sum()
df.groupby('col').mean()
df.groupby('col').agg({'col2': 'sum', 'col3': 'mean'})

# Ordenar
df.sort_values('col', ascending=False)

# Tipos de datos
df.astype({'col': int, 'col2': float})
```

---

## 🔟 NUMPY ESSENTIALS

```python
import numpy as np

# Crear arrays
np.array([1, 2, 3])
np.zeros((3, 3))
np.ones((3, 3))
np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]

# Operaciones
arr * 2          # Elemento a elemento
arr + arr        # Suma elemento a elemento
arr.sum()        # Suma total
arr.mean()       # Media
arr.std()        # Desviación estándar
arr.max()        # Máximo
arr.reshape(2, 3)  # Cambiar forma

# Indexing
arr[0]           # Primera elemento
arr[1:3]         # Índices 1 y 2
arr[arr > 5]     # Elementos > 5
```

---

## 🎯 CHECKLIST PARA EL EXAMEN

- [ ] ¿Cuál es el problema? Regresión/Clasificación/Clustering
- [ ] ¿Cuáles son las features (X)? ¿Cuál es el target (y)?
- [ ] ¿Hay valores faltantes? Limpiar
- [ ] ¿Están normalizados? (si es necesario)
- [ ] ¿Train/Test split? (80/20 típico)
- [ ] ¿Qué modelo es apropiado?
- [ ] ¿Entrenar y evaluar?
- [ ] ¿Qué métricas son relevantes?
- [ ] ¿Interpretar resultados?

---

## ⚡ TRUCOS RÁPIDOS

```python
# Ver todas las columnas en pandas
pd.set_option('display.max_columns', None)

# Ver todas las filas
pd.set_option('display.max_rows', None)

# Seed para reproducibilidad
import random
random.seed(42)
np.random.seed(42)

# Verificar si modelo está overfitting
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
if train_score >> test_score:
    print("OVERFITTING DETECTADO")
```

---

## 📚 REFERENCIA DE IMPORTS

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Modelos
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans

# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score

# Métricas
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, silhouette_score
)

# Pipeline
from sklearn.pipeline import Pipeline
```

---

**Última actualización:** 12 Mayo 2026
**Examen:** 19 Mayo 2026 (Semana que viene)

