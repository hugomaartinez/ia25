# 🎯 Soluciones Completas - Todos los Tipos de Ejercicios

---

# PARTE 1: EJERCICIOS PRÁCTICOS CON SOLUCIONES

---

## TIPO 1: PROBLEMA DE REGRESIÓN

### Enunciado Típico:
"Tienes un dataset con características de viviendas (metros², habitaciones, año construcción). Predice el precio de venta."

### Solución Paso a Paso:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# PASO 1: CARGAR Y EXPLORAR
df = pd.read_csv('viviendas.csv')
print(df.head())
print(df.info())           # Ver tipos de datos
print(df.describe())       # Estadísticas
print(df.isnull().sum())   # Valores faltantes

# PASO 2: LIMPIAR
# Si hay faltantes:
df = df.fillna(df.mean())  # Llenar con media (para numéricos)
# O: df = df.dropna()      # Eliminar filas con NaN

# Si hay columnas categóricas:
df = pd.get_dummies(df, columns=['tipo_zona'])  # One-hot encoding

# PASO 3: SEPARAR FEATURES Y TARGET
X = df.drop('precio', axis=1)  # Features
y = df['precio']               # Target (lo que queremos predecir)

# PASO 4: DIVIDIR EN TRAIN/TEST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# PASO 5: NORMALIZAR (IMPORTANTE para regresión lineal)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PASO 6: ENTRENAR MODELO
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# PASO 7: PREDECIR
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# PASO 8: EVALUAR
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"MSE Train: {mse_train:.2f}, Test: {mse_test:.2f}")
print(f"RMSE Test: {rmse_test:.2f}")
print(f"MAE Test: {mae_test:.2f}")
print(f"R² Test: {r2_test:.4f}")

# PASO 9: DETECTAR OVERFITTING
if mse_train << mse_test:  # Si train << test
    print("⚠️ POSIBLE OVERFITTING")
else:
    print("✅ Modelo generaliza bien")

# PASO 10: ALTERNATIVA - Random Forest (mejor para relaciones no lineales)
model_rf = RandomForestRegressor(n_estimators=100, max_depth=10)
model_rf.fit(X_train, y_train)  # NO necesita normalización
y_pred_rf = model_rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"R² Random Forest: {r2_rf:.4f}")
```

### Checklist para regresión:
- ✅ Explora datos primero (EDA)
- ✅ Maneja faltantes apropiadamente
- ✅ Normaliza ANTES de train/test split
- ✅ Usa scaler.fit(X_train), luego transform(X_test)
- ✅ Compara error train vs test
- ✅ Reporta: MSE, RMSE, MAE, R²
- ✅ Interpreta: "R²=0.85 significa el modelo explica 85% de varianza"

---

## TIPO 2: PROBLEMA DE CLASIFICACIÓN

### Enunciado Típico:
"Dataset de iris: clasifica en 3 tipos. Evalúa el modelo."

### Solución Paso a Paso:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
import seaborn as sns
import matplotlib.pyplot as plt

# PASO 1: CARGAR DATOS
iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

# PASO 2: EXPLORACIÓN
print(df.value_counts('target'))  # Ver distribución de clases
# Si hay desbalance (ej: 90% clase 0, 10% clase 1) → problema

# PASO 3: DIVIDIR
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify es IMPORTANTE
)

# PASO 4: NORMALIZAR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PASO 5: ENTRENAR
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

# PASO 6: PREDECIR
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)  # Probabilidades

# PASO 7: EVALUAR - MATRIZ DE CONFUSIÓN
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")
# Para binaria:
# TN (verdadero neg) | FP (falso pos)
# FN (falso neg)      | TP (verdadero pos)

# PASO 8: EVALUAR - MÉTRICAS
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # Para multiclase
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy:  {acc:.4f}")      # Proporción total correcta
print(f"Precision: {precision:.4f}") # De mis predicciones +, cuántas son correctas
print(f"Recall:    {recall:.4f}")    # De los + reales, cuántos detecté
print(f"F1-Score:  {f1:.4f}")        # Balance precision-recall

# PASO 9: REPORTE DETALLADO
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# PASO 10: VISUALIZAR MATRIZ DE CONFUSIÓN
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Real')
plt.xlabel('Predicho')
plt.show()

# PASO 11: PROBLEMA BINARIO ESPECIAL
# Si es binaria, puedes ver la matriz así:
if len(np.unique(y)) == 2:
    tn, fp, fn, tp = cm.ravel()
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    sensitivity = tp / (tp + fn)  # Recall
    specificity = tn / (tn + fp)
    print(f"Sensitivity: {sensitivity}, Specificity: {specificity}")
```

### Decisiones importantes:
- **¿Accuracy es suficiente?** 
  - ✅ SÍ si clases balanceadas
  - ❌ NO si desbalanceadas (ej: 95% clase 0)
  
- **¿Precision vs Recall?**
  - Precision: Evitar falsos positivos (spam filter, videos seguros)
  - Recall: Detectar todos los positivos (diagnósticos médicos, fraude)

---

## TIPO 3: PROBLEMA DE CLUSTERING

### Enunciado Típico:
"Agrupa 200 clientes en 3 grupos sin etiquetar. Evalúa."

### Solución Paso a Paso:

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

# PASO 1: PREPARAR DATOS
X = df[['feature1', 'feature2', 'feature3']].values

# PASO 2: NORMALIZAR (CRÍTICO para clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PASO 3: ENCONTRAR K ÓPTIMO (Elbow Method)
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Graficar Elbow
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('K (número de clusters)')
plt.ylabel('Inercia')
plt.title('Elbow Method')
plt.grid()

# Graficar Silhouette
plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, 'go-')
plt.xlabel('K')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score por K')
plt.grid()
plt.show()

# El "codo" es el K óptimo (punto de inflexión)

# PASO 4: ENTRENAR CON K ÓPTIMO
k_optimal = 3  # O lo que encuentres del gráfico
kmeans = KMeans(n_clusters=k_optimal, n_init=20, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# PASO 5: OBTENER INFORMACIÓN
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
inertia = kmeans.inertia_

# PASO 6: EVALUAR
silhouette = silhouette_score(X_scaled, clusters)
davies_bouldin = davies_bouldin_score(X_scaled, clusters)

print(f"Silhouette Score: {silhouette:.4f}")  # Rango -1 a 1, más alto mejor
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")  # Más bajo mejor
print(f"Inercia: {inertia:.2f}")

# PASO 7: INTERPRETAR CLUSTERS
for cluster_id in range(k_optimal):
    cluster_data = df[clusters == cluster_id]
    print(f"\nCluster {cluster_id}:")
    print(f"  Tamaño: {len(cluster_data)} puntos")
    print(f"  Características promedio:\n{cluster_data.describe().loc['mean']}")

# PASO 8: VISUALIZAR (si 2 o 3D)
if X_scaled.shape[1] >= 2:
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', s=50)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'K-Means Clustering (K={k_optimal})')
    plt.colorbar(scatter)
    plt.show()
```

### Checklist clustering:
- ✅ Normaliza SIEMPRE (StandardScaler)
- ✅ Prueba múltiples K (2-10)
- ✅ Usa Elbow Method o Silhouette Score
- ✅ Evalúa con Silhouette (-1 a 1, más alto mejor)
- ✅ Interpreta clusters (tamaño, características)
- ✅ Visualiza si es posible

---

## TIPO 4: PROBLEMA DE VALIDACIÓN/OVERFITTING

### Enunciado Típico:
"Entrena un árbol. ¿Hay overfitting? Justifica."

### Solución:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# OPCIÓN 1: Comparar Train vs Test
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Train Accuracy: {train_score:.4f}")
print(f"Test Accuracy:  {test_score:.4f}")

if train_score - test_score > 0.1:  # Diferencia > 10%
    print("⚠️ PROBABLE OVERFITTING")
    # Soluciones:
    # 1. Reducir max_depth
    # 2. Aumentar min_samples_split
    # 3. Agregar más datos de entrenamiento
else:
    print("✅ Generalización aceptable")

# OPCIÓN 2: Curva de Aprendizaje
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train, cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.plot(train_sizes, train_mean, label='Train')
plt.plot(train_sizes, val_mean, label='Validation')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Learning Curve')
plt.show()

# Si train sube pero val baja → OVERFITTING

# OPCIÓN 3: Validación Cruzada
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Scores: {scores}")
print(f"Media: {scores.mean():.4f}, Std: {scores.std():.4f}")
# Si std alto → modelo inestable
```

---

## TIPO 5: PROBLEMA DE MANEJO DE DATOS FALTANTES

### Enunciado Típico:
"Dataset tiene 15% de valores faltantes en columna 'edad'. ¿Qué haces?"

### Soluciones:

```python
# OPCIÓN 1: VER PATRÓN DE FALTANTES
import pandas as pd
print(df.isnull().sum())
print(df.isnull().sum() / len(df) * 100)  # Porcentaje

# OPCIÓN 2: SIMPLE - ELIMINAR
df_clean = df.dropna()  # Elimina filas con CUALQUIER NaN
df_clean = df.dropna(subset=['edad'])  # Solo en columna 'edad'

# RIESGO: Pierdes datos, posible sesgo

# OPCIÓN 3: IMPUTAR CON MEDIA
df['edad'].fillna(df['edad'].mean(), inplace=True)

# O con mediana (más robusto a outliers):
df['edad'].fillna(df['edad'].median(), inplace=True)

# OPCIÓN 4: IMPUTAR CON KNN
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df[['edad', 'ingreso']] = imputer.fit_transform(df[['edad', 'ingreso']])

# OPCIÓN 5: IMPUTAR CON ÁRBOL (más sofisticado)
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(random_state=42)
df[['edad', 'ingreso']] = imputer.fit_transform(df[['edad', 'ingreso']])

# OPCIÓN 6: CREAR VARIABLE INDICADORA
df['edad_missing'] = df['edad'].isnull().astype(int)
df['edad'].fillna(df['edad'].mean(), inplace=True)
# Ahora tienes columna que indica qué estaba faltante

print("✅ Datos limpios")
print(df.isnull().sum())  # Verificar
```

---

## TIPO 6: PIPELINE CON SKLEARN

### Enunciado Típico:
"Crea un pipeline que normalice y clasifique. Evita data leakage."

### Solución:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# CREAR PIPELINE
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=200))
])

# DIVIDIR DATOS
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ENTRENAR (Pipeline automáticamente hace fit_transform en train)
pipeline.fit(X_train, y_train)

# PREDECIR (Pipeline automáticamente hace transform en test)
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

# EVALUAR
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print(f"Train: {train_acc:.4f}, Test: {test_acc:.4f}")

# ✅ VENTAJA: No hay data leakage (scaler solo vio X_train)
# ❌ SIN PIPELINE: 
#    scaler.fit(X)  # ❌ VE TODO INCLUYENDO TEST
#    scaler.transform(X_train)
#    scaler.transform(X_test)
```

---

# PARTE 2: PREGUNTAS TEÓRICAS MÁS IMPORTANTES

---

## TEMA 1: MÉTRICAS DE EVALUACIÓN

### P1: ¿Cuál es la diferencia entre Accuracy y Precision?

**Accuracy:** Proporción de TODAS las predicciones correctas  
Formula: (TP + TN) / Total

**Precision:** De mis predicciones POSITIVAS, cuántas son correctas  
Formula: TP / (TP + FP)

**Ejemplo:** Filtro de spam  
- Accuracy: ¿Cuántos emails clasifiqué correctamente? (mayormente no-spam)
- Precision: ¿De los que marqué como spam, cuántos realmente eran spam?
- Si precision es baja, bloqueo muchos emails legítimos (malo)

---

### P2: ¿Cuándo usar Recall en lugar de Precision?

**Recall** = TP / (TP + FN) = ¿De los positivos reales, cuántos detecté?

**Cuándo prioritariamente Recall:**
- Diagnóstico médico (no puedes dejar pasar cáncer)
- Detección de fraude (mejor falsa alarma que perder fraude)
- Seguridad (mejor paranoico que inseguro)

**Cuándo prioritariamente Precision:**
- Spam filter (mejor dejar pasar spam que bloquear legítimo)
- Videos seguros para niños (mejor dejar pasar dudoso que bloquear seguro)

---

### P3: ¿Qué es F1-Score y cuándo se usa?

**F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)

- Rango: 0 a 1 (1 es perfecto)
- Es el **balance** entre Precision y Recall
- **Cuándo usarlo:** Cuando ambas métricas son importantes y quieres un número único

**Ejemplo:** Clasificador de enfermedades
- Precision: No diagnosticar falsamente
- Recall: No dejar sin tratar
- F1: Balance de ambas

---

### P4: ¿Por qué Accuracy es engañosa con clases desbalanceadas?

**Ejemplo:** Dataset de detección de fraude
- 99% transacciones legítimas
- 1% fraudulentas

**Modelo ingenuo:**
```python
# Modelo que predice "legítimo" SIEMPRE
accuracy = 0.99  # ¡Parece excelente!
recall = 0        # ¡Detecta 0 fraudes!
```

**Problema:** Accuracy es alto pero modelo es inútil

**Solución:** Usar Precision, Recall, F1 o Confusion Matrix

---

### P5: ¿Qué es la Matriz de Confusión?

```
                Predicho
              Positivo  Negativo
Real Positivo   TP        FN
     Negativo   FP        TN
```

- **TP (Verdadero Positivo):** Predicho +, real + → Correcto
- **TN (Verdadero Negativo):** Predicho -, real - → Correcto
- **FP (Falso Positivo):** Predicho +, real - → Error Tipo I
- **FN (Falso Negativo):** Predicho -, real + → Error Tipo II

---

### P6: ¿MSE vs RMSE vs MAE? ¿Cuándo usar cada una?

| Métrica | Fórmula | Penaliza | Cuándo usar |
|---------|---------|----------|------------|
| MSE | (Σ(y-ŷ)²)/n | Errores grandes | Optimización |
| RMSE | √MSE | Errores grandes | Misma unidad que y |
| MAE | Σ\|y-ŷ\|/n | Todos igual | Robusto a outliers |
| R² | 1-(SS_res/SS_tot) | N/A | Proporción varianza |

**Ejemplo:** Predicción de precios (rango 100k-500k)
- MSE = 50,000,000 → Difícil de interpretar
- RMSE = 7,071 → "Error promedio: $7,071"
- MAE = 5,000 → "Error promedio: $5,000"
- R² = 0.85 → "Explico 85% de variabilidad"

---

### P7: ¿Qué es Silhouette Score en clustering?

**Rango:** -1 a +1

- **+1:** Clusters densos, bien separados
- **0:** Clusters solapados
- **-1:** Puntos asignados al cluster equivocado

**Uso:** Evaluar calidad del clustering

---

## TEMA 2: SUPERVISADO VS NO SUPERVISADO

### P8: ¿Diferencia fundamental?

**SUPERVISADO:** Tienes etiquetas (target/y)
- Regresión: Predicción de números continuos
- Clasificación: Predicción de categorías

**NO SUPERVISADO:** Sin etiquetas
- Clustering: Agrupar similares
- Dimensionalidad: Reducir features
- Anomaly Detection: Encontrar outliers

**REFUERZO:** Agente aprende interactuando
- Q-Learning, DQN
- Markov Decision Process

---

### P9: ¿Cuándo usar cada paradigma?

| Tengo... | Uso | Ejemplo |
|----------|-----|---------|
| Datos con etiquetas | Supervisado | Iris con etiquetas |
| Datos sin etiquetas, quiero agrupar | Clustering | Segmentación de clientes |
| Datos sin etiquetas, quiero reducir features | PCA | Visualizar 100 features en 2D |
| Datos sin etiquetas, quiero detectar anomalías | Anomaly Detection | Fraude bancario |
| Agente interactúa con entorno | Refuerzo | Juegos, robots |

---

## TEMA 3: OVERFITTING Y GENERALIZACIÓN

### P10: ¿Cómo detectar overfitting?

**Síntoma:** error_train << error_test

**Ejemplo:**
```
Train Accuracy: 0.99
Test Accuracy:  0.65
→ OVERFITTING CLARO
```

**Método 1:** Comparar train vs test
**Método 2:** Curva de aprendizaje (train sube, validación baja)
**Método 3:** Cross-validation (alta varianza en scores)

---

### P11: ¿Cómo prevenir overfitting?

| Problema | Causa | Solución |
|----------|-------|----------|
| Modelo muy complejo | Demasiados parámetros | Reducir max_depth, aumentar min_samples |
| Pocos datos | Memoriza | Agregar datos |
| Entrenamiento largo | Se ajusta a ruido | Early stopping |
| N/A | N/A | Regularización (L1, L2) |

---

### P12: ¿Qué es underfitting?

**Síntoma:** error_train ≈ error_test (ambos altos)

**Causa:** Modelo muy simple

**Solución:** Modelo más complejo (más parámetros, más capas)

---

## TEMA 4: DATOS FALTANTES

### P13: ¿Cuál es la diferencia entre MCAR, MAR y MNAR?

**MCAR (Missing Completely At Random)**
- Causa: Puramente aleatoria
- Ejemplo: Error aleatorio del sensor
- Solución: Borrar es OK

**MAR (Missing At Random)**
- Causa: Depende de variables observadas
- Ejemplo: Gente con ingresos bajos más probable que omita edad
- Solución: Imputar es mejor

**MNAR (Missing Not At Random)**
- Causa: En el propio valor faltante
- Ejemplo: Personas con enfermedades evitan reportar síntomas
- Solución: Difícil, puede haber sesgo

---

### P14: ¿Riesgos de eliminar filas con faltantes?

1. **Pérdida de información:** Otras columnas de esa fila también se pierden
2. **Sesgo:** Si no MCAR, puede sesgarse el análisis
3. **Reducción de datos:** Menos muestras para entrenar

---

### P15: ¿Cuál es mejor método de imputación?

| Método | Ventajas | Desventajas |
|--------|----------|-------------|
| Media | Rápido | Sesga varianza |
| Mediana | Robusto a outliers | Sesga varianza |
| KNN | Preserva correlaciones | Computacionalmente costoso |
| Árbol iterativo | Más realista | Complejo, lento |

**Recomendación:** Mediana para datos normales, KNN si hay correlaciones

---

## TEMA 5: NORMALIZACIÓN Y ESCALADO

### P16: ¿StandardScaler vs MinMaxScaler?

**StandardScaler:** x' = (x - mean) / std
- Rango: Indefinido
- Menos sensible a outliers

**MinMaxScaler:** x' = (x - min) / (max - min)
- Rango: [0, 1]
- MUY sensible a outliers (el rango se estira)

**Cuándo usar:**
- StandardScaler: Modelos lineales, SVM, redes neuronales
- MinMaxScaler: Cuando necesitas [0, 1] explícitamente

---

### P17: ¿Cuál es el error común con escalado?

**INCORRECTO:**
```python
scaler.fit(X)  # Calcula media/std con TODO incluyendo TEST
scaler.transform(X_train)
scaler.transform(X_test)
# → DATA LEAKAGE
```

**CORRECTO:**
```python
scaler.fit(X_train)  # Solo aprende de TRAIN
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# → Sin leakage
```

---

## TEMA 6: ALGORITMOS

### P18: ¿Regresión Lineal vs Logística?

**Regresión Lineal:**
- Output: Números continuos (sin límite)
- Formula: y = β₀ + β₁X + ε
- Usa: Predecir precios, temperaturas

**Regresión Logística:**
- Output: Probabilidades [0, 1]
- Formula: P(y=1|X) = sigmoide(β₀ + β₁X)
- Usa: Clasificación binaria

---

### P19: ¿Cuándo usar KNN?

**Ventajas:**
- Simple conceptualmente
- No hace supuestos

**Desventajas:**
- Predicción LENTA (compara contra todos)
- Sensible a outliers
- Necesita normalización

**Cuándo usar:** Solo para datasets pequeños (<10k ejemplos)

---

### P20: ¿Árboles de Decisión - cuándo entra en overfitting?

**Sin límites:** El árbol crece para encajar perfectamente → OVERFITTING

**Prevención:**
- `max_depth`: Profundidad máxima
- `min_samples_split`: Muestras mínimas para dividir
- `min_samples_leaf`: Muestras mínimas en hoja

**Ventaja:** Nativo con datos faltantes (aprende dirección)

---

### P21: ¿Random Forest vs Árbol Individual?

**Random Forest:**
- Múltiples árboles con bootstrap (con reemplazo)
- Promedia predicciones (reduce varianza)
- Menos overfitting

**Árbol Individual:**
- Puede crecer demasiado
- Mayor overfitting
- Menos robusto

**Ganador:** Random Forest casi siempre

---

### P22: ¿K-Means - qué minimiza?

**Minimiza:** Inercia (WCSS)
= Suma de distancias al cuadrado de cada punto a su centroide

**Sensibilidad:** Sensible a inicialización → usar `n_init=20` para múltiples ejecuciones

---

## TEMA 7: DATA LEAKAGE

### P23: ¿Qué es Data Leakage?

**Definición:** Información del TEST set filtra al entrenamiento

**Ejemplos comunes:**
1. Normalizar antes de dividir
2. Seleccionar features usando TODO el dataset
3. Usar test set para ajustar hiperparámetros

**Consecuencia:** Métricas optimistas, rendimiento real es peor

---

### P24: ¿Qué reglas de negocio SÍ pueden usarse antes del split?

**SÍ - Son universales:**
- Edad > 150 es error → Filtrar
- Precio negativo es error → Remover
- Formato de email inválido → Remover

**NO - Son estadísticas:**
- Eliminar outliers usando percentiles
- Llenar faltantes con media
- Normalizar

---

## TEMA 8: VALIDACIÓN

### P25: ¿Train/Test/Validation Split?

```
TRAIN (60-70%):     Entrenar pesos
VALIDATION (15%):   Ajustar hiperparámetros durante entrenamiento
TEST (15-20%):      Evaluar ÚNICO (toca una sola vez)
```

---

### P26: ¿Validación Cruzada?

**Qué hace:**
- Divide train en K folds
- Entrena K veces, cada vez usando diferentes fold para validación
- Promedia resultados

**Ventajas:** Usa todos los datos, reduce varianza de estimación
**Desventajas:** K veces más costoso computacionalmente

**Cuándo usar:** Datasets pequeños (<10k)

---

### P27: ¿Cross-Validation Strategy?

```python
scores = cross_val_score(model, X, y, cv=5)
# [0.92, 0.88, 0.90, 0.91, 0.89]

mean = scores.mean()      # 0.90
std = scores.std()        # 0.013

# Si std alto → modelo inestable
# Si std bajo → modelo estable
```

---

## TEMA 9: PIPELINES

### P28: ¿Ventaja principal de Pipelines?

**Previene Data Leakage automáticamente**

```python
# SIN PIPELINE (riesgoso)
scaler.fit(X_train)      # ← Manual, fácil equivocarse
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# CON PIPELINE (seguro)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipe.fit(X_train, y_train)  # Automáticamente fit_transform en train
pipe.predict(X_test)        # Automáticamente transform en test
```

---

### P29: ¿Requisitos de Pasos en Pipeline?

**Pasos intermedios:** Deben ser TRANSFORMADORES
- Implementar `fit()` y `transform()`
- Ejemplos: StandardScaler, OneHotEncoder, PCA

**Paso final:** Puede ser ESTIMADOR
- Solo necesita `fit()` y `predict()`
- Ejemplos: LogisticRegression, DecisionTree

---

## TEMA 10: DEEP LEARNING BASICS

### P30: ¿Qué es un Perceptrón?

**Definición:** Unidad básica de red neuronal
- Entradas ponderadas (X₁, X₂, ...)
- Suma ponderada (ΣwᵢXᵢ + b)
- Función de activación (ReLU, sigmoid)

**Limitación:** Solo resuelve problemas linealmente separables
- Ejemplo: No puede aprender XOR

---

### P31: ¿Diferencia entre Perceptrón y MLP?

**Perceptrón:** Una sola capa
- Solo problemas lineales

**MLP (Multi-Layer Perceptron):** Múltiples capas
- Capas ocultas introducen no-linealidad
- Puede aprender funciones complejas

---

### P32: ¿Backpropagation - concepto?

**Proceso:**
1. Forward pass: Datos pasan adelante, calcular predicción
2. Calcular error (predicción vs real)
3. Backward pass: Propagar error hacia atrás
4. Actualizar pesos en dirección que reduce error

**No necesitas:** Fórmulas detalladas, solo concepto

---

### P33: ¿ReLU vs Sigmoid vs Tanh?

**ReLU:** max(0, x)
- Rango: [0, ∞)
- Uso: Capas ocultas (estándar)

**Sigmoid:** 1/(1+e^(-x))
- Rango: [0, 1]
- Uso: Output clasificación binaria

**Tanh:** (e^x - e^(-x))/(e^x + e^(-x))
- Rango: [-1, 1]
- Uso: Capas ocultas (alternativa a ReLU)

---

### P34: ¿Cross-Entropy vs MSE?

**Cross-Entropy:**
- Para clasificación
- Penaliza predicciones confidentes pero incorrectas

**MSE:**
- Para regresión
- Penaliza diferencia al cuadrado

---

## TEMA 11: REINFORCEMENT LEARNING (Q-LEARNING)

### P35: ¿Qué es Q-Learning?

**Idea:** Agente aprende tabla Q(estado, acción) que dice la recompensa futura esperada

**Componentes:**
- Estado: Posición actual en el entorno
- Acción: Movimiento disponible
- Recompensa: Retroalimentación
- Q-value: Recompensa futura esperada

---

### P36: ¿Epsilon-Greedy?

**Balance exploración vs explotación:**
- Con probabilidad ε: Acción aleatoria (explorar)
- Con probabilidad 1-ε: Mejor acción conocida (explotar)

**ε decay:** ε empieza alto (mucha exploración), disminuye (más explotación)

---

### P37: ¿Learning Rate en Q-Learning?

**α (alpha):**
- α=1: Olvida pasado, usa solo información nueva (inestable)
- α=0: No aprende nada
- α=0.1-0.5: Balance típico

---

### P38: ¿Discount Factor?

**γ (gamma):**
- γ=0: Solo recompensa inmediata importa
- γ=1: Recompensa futura es muy importante
- γ=0.9-0.99: Típico (se importa el futuro pero también presente)

---

## PREGUNTAS TIPO TRAMPA

### P39: "¿Accuracy es la mejor métrica?"

**INCORRECTO.** Depende:
- Accuracy: Solo si clases balanceadas
- Precision/Recall: Si clases desbalanceadas
- F1: Si quieres balance

---

### P40: "¿Normalización es opcional?"

**INCORRECTO.** Es crítico para:
- Modelos basados en distancia (KNN, SVM, K-Means)
- Regresión lineal con regularización
- Redes neuronales

---

### P41: "¿Puedo usar test set para ajustar hiperparámetros?"

**INCORRECTO.** Causaría data leakage.
- Usa validation set o cross-validation

---

### P42: "¿El modelo con mayor accuracy en train es el mejor?"

**INCORRECTO.** Probablemente overfitting.
- Compara train vs test
- Si diferencia grande → overfitting

---

