# 💻 PRÁCTICA - CÓDIGO Y EJERCICIOS (Units 1-10)

**Fecha:** 12 Mayo 2026  
**Examen Práctico:** 26 Mayo 2026  
**Todos los snippets son copy-paste ready**

---

## 🟢 WORKFLOW ESTÁNDAR (TODOS LOS PROBLEMAS)

```python
# 1. IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. CARGAR DATOS
df = pd.read_csv('data.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())  # Datos faltantes

# 3. EXPLORACIÓN (EDA)
# - Distribución de variables
# - Correlaciones
# - Outliers
# - Desbalance de clases

# 4. PREPROCESAMIENTO
# - Datos faltantes: imputar o borrar
# - Variables categóricas: get_dummies()
# - Outliers: IQR o Z-score
# - Normalizació: StandardScaler

# 5. SPLIT
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. NORMALIZACIÓN (fit TRAIN, transform AMBOS)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. MODELO Y ENTRENAR
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 8. PREDICCIÓN
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probabilidades

# 9. EVALUACIÓN
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 10. VISUALIZACIÓN
plt.hist(y_pred_proba, bins=30)
plt.xlabel('Probabilidad')
plt.ylabel('Frecuencia')
plt.show()
```

---

## 📊 TIPO 1: REGRESIÓN

### Problema: Predecir precio de casa (y continua)

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Datos
X_train = [[100], [200], [300], [400], [500]]
y_train = [10000, 20000, 30000, 40000, 50000]
X_test = [[150], [250], [350]]
y_test = [15000, 25000, 35000]

# Modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Predicción
y_pred = model.predict(X_test)

# Evaluación
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE:  {mse}")
print(f"RMSE: {rmse}")
print(f"MAE:  {mae}")
print(f"R²:   {r2}")

# Interpretación
print(f"Coeficiente: {model.coef_[0]}")  # Por cada unidad X, y aumenta esto
print(f"Intercept:   {model.intercept_}")  # Valor base
```

**Análisis:**
- Si R² < 0.5: Modelo malo
- Si 0.5 < R² < 0.7: Modelo regular
- Si R² > 0.7: Modelo bueno
- RMSE en misma unidad que y (interpretable)
- MAE más robusto a outliers

---

## 📊 TIPO 2: CLASIFICACIÓN BINARIA

### Problema: ¿Spam o No-Spam?

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Datos
X_train = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
y_train = [0, 0, 0, 1, 1, 1]
X_test = [[1.5, 2.5], [4.5, 5.5]]
y_test = [0, 1]

# Modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# Predicción
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluación
cm = confusion_matrix(y_test, y_pred)
#         pred_no  pred_yes
# real_no   TN       FP
# real_yes  FN       TP

tn, fp, fn, tp = cm.ravel()
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f"Accuracy:  {accuracy}")
print(f"Precision: {precision}  (TP/(TP+FP))")
print(f"Recall:    {recall}     (TP/(TP+FN))")
print(f"F1:        {f1}")

# Interpretación
if precision > recall:
    print("→ Prefiero NO marcar falsos positivos (confianza)")
else:
    print("→ Prefiero detectar TODOS los positivos (cobertura)")
```

---

## 📊 TIPO 3: CLASIFICACIÓN MULTICLASE

### Problema: Iris (setosa, versicolor, virginica)

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Datos
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo Random Forest (mejor para multiclase)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Predicción
y_pred = model.predict(X_test)

# Evaluación
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Importancia de features
for i, importance in enumerate(model.feature_importances_):
    print(f"Feature {i}: {importance:.4f}")
```

---

## 📊 TIPO 4: CLUSTERING

### Problema: Agrupar clientes sin etiquetas

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# Datos
X = np.array([
    [1, 2], [2, 3], [3, 4],
    [10, 11], [11, 12], [12, 13]
])

# Elbow Method (elegir K)
inertias = []
silhouette_scores = []
K_range = range(2, 6)

for k in K_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Gráfica (busca "codo" en inertia)
import matplotlib.pyplot as plt
plt.plot(K_range, inertias, 'o-')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.show()

# Mejor K según silhouette (más cercano a 1)
best_k = K_range[np.argmax(silhouette_scores)]

# Modelo final
kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

print(f"Mejor K: {best_k}")
print(f"Silhouette Score: {silhouette_score(X, labels):.3f}")
print(f"Inertia: {kmeans.inertia_:.3f}")
print(f"Centroides:\n{centers}")
```

---

## 📊 TIPO 5: MANEJO DATOS FALTANTES

### Problema: Dataset con valores NaN

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

# Datos con faltantes
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [10, np.nan, 30, 40, 50],
    'C': [100, 200, 300, 400, 500]
})

print("Datos originales:")
print(df)
print("\nFaltantes por columna:")
print(df.isnull().sum())

# OPCIÓN 1: Borrar (si MCAR)
df_dropped = df.dropna()
print("\nDespués de dropna:")
print(df_dropped)

# OPCIÓN 2: Imputar con media
imputer_mean = SimpleImputer(strategy='mean')
X_imputed = imputer_mean.fit_transform(df[['A', 'B']])
df['A'] = X_imputed[:, 0]
df['B'] = X_imputed[:, 1]
print("\nDespués de imputar media:")
print(df)

# OPCIÓN 3: Imputar con KNN (más avanzado)
imputer_knn = KNNImputer(n_neighbors=3)
X_imputed = imputer_knn.fit_transform(df)
print("\nDespués de imputar KNN:")
print(X_imputed)

# Decisión:
# - MCAR: Borrar es OK
# - MAR: Imputar con media/KNN
# - MNAR: Difícil, riesgo sesgo
```

---

## 📊 TIPO 6: NORMALIZACIÓN (CRÍTICO)

### Problema: Features con escalas diferentes

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Datos
X = [[1, 1000], [2, 2000], [3, 3000], [4, 4000]]
y = [0, 0, 1, 1]

# SPLIT PRIMERO
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# ✅ CORRECTO: fit TRAIN, transform AMBOS
print("=== CORRECTO ===")
scaler = StandardScaler()
scaler.fit(X_train)  # Calcula mean, std SOLO en train
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)   # Usa parámetros de train
print(f"Train escalado:\n{X_train_scaled}")
print(f"Test escalado:\n{X_test_scaled}")

# ❌ INCORRECTO: fit TEST (data leakage)
print("\n=== INCORRECTO ===")
scaler_bad = StandardScaler()
scaler_bad.fit(X_test)  # ❌ NO HACER ESTO
X_test_bad = scaler_bad.transform(X_test)
print(f"Test escalado (mal):\n{X_test_bad}")  # Test afecta train

# Comparación StandardScaler vs MinMaxScaler
print("\n=== StandardScaler vs MinMaxScaler ===")
ss = StandardScaler()
mms = MinMaxScaler()

ss.fit(X_train)
mms.fit(X_train)

X_train_ss = ss.transform(X_train)
X_train_mms = mms.transform(X_train)

print(f"StandardScaler (unbounded):\n{X_train_ss}")
print(f"MinMaxScaler [0,1]:\n{X_train_mms}")

# Con outlier
X_with_outlier = [[1, 1000], [2, 2000], [3, 3000], [4, 4000], [5, 10000]]
mms2 = MinMaxScaler()
mms2.fit(X_with_outlier)
print(f"\nMinMaxScaler CON outlier (se estira):\n{mms2.transform(X_with_outlier)}")
# ^ Nota: El rango [0,1] se estira por el outlier
```

---

## 📊 TIPO 7: DETECCIÓN OVERFITTING

### Problema: Modelo memoriza en lugar de generalizar

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

# Datos
X = np.random.randn(100, 10)
y = np.random.binomial(1, 0.5, 100)

# Curva de aprendizaje (detecta overfitting)
train_sizes, train_scores, val_scores = learning_curve(
    DecisionTreeClassifier(max_depth=20),
    X, y,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.plot(train_sizes, train_mean, label='Train')
plt.plot(train_sizes, val_mean, label='Validation')
plt.xlabel('Training size')
plt.ylabel('Score')
plt.legend()
plt.show()

# Interpretación:
# - Train sigue bajando, Val sube → OVERFITTING (modelo muy complejo)
# - Train y Val altos → BUENO
# - Train y Val ambos bajos → UNDERFITTING (modelo simple)

# Métricas explícitas
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model_simple = DecisionTreeClassifier(max_depth=3)
model_complex = DecisionTreeClassifier(max_depth=20)

model_simple.fit(X_train, y_train)
model_complex.fit(X_train, y_train)

print("SIMPLE (max_depth=3):")
print(f"  Train: {model_simple.score(X_train, y_train):.3f}")
print(f"  Test:  {model_simple.score(X_test, y_test):.3f}")
print(f"  Diferencia: {model_simple.score(X_train, y_train) - model_simple.score(X_test, y_test):.3f}")

print("\nCOMPLEX (max_depth=20):")
print(f"  Train: {model_complex.score(X_train, y_train):.3f}")
print(f"  Test:  {model_complex.score(X_test, y_test):.3f}")
print(f"  Diferencia: {model_complex.score(X_train, y_train) - model_complex.score(X_test, y_test):.3f}")

# Soluciones overfitting:
# 1. Reducir max_depth (más simple)
# 2. Aumentar min_samples_leaf
# 3. Más datos
# 4. Regularización (Ridge, Lasso)
# 5. Validación cruzada
```

---

## 🔵 UNIT 6: DEEP LEARNING - PYTORCH

### PyTorch Básico: MLP en MNIST

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DATA
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# MODEL: MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)      # 28*28=784 → 128
        self.fc2 = nn.Linear(128, 64)       # 128 → 64
        self.fc3 = nn.Linear(64, 10)        # 64 → 10 (clases)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)           # Flatten
        x = torch.relu(self.fc1(x))         # ReLU
        x = torch.relu(self.fc2(x))         # ReLU
        x = self.fc3(x)                     # Output (sin activación, lo hace CrossEntropy)
        return x

model = MLP().to(device)

# LOSS & OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TRAINING
epochs = 5
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward
        outputs = model(data)
        loss = criterion(outputs, target)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

# EVALUATION
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == target).sum().item()
        total += target.size(0)

accuracy = 100 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%")
```

---

## 🔵 UNIT 7: CNN

### CNN en CIFAR-10

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Convoluciones
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Capas densas
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # Conv → ReLU → Pool
        x = self.pool(torch.relu(self.conv1(x)))  # 32x16x16
        x = self.pool(torch.relu(self.conv2(x)))  # 64x8x8
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop igual al anterior
```

---

## 🔵 UNIT 8: TRANSFER LEARNING

### ResNet preentrenado

```python
from torchvision import models

# Cargar ResNet preentrenado
model = models.resnet18(pretrained=True)

# Congelar pesos iniciales
for param in model.parameters():
    param.requires_grad = False

# Reemplazar última capa
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)  # 10 clases nuevas

# Solo entrena la última capa
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Training normal (solo actualiza fc)
# model.fc(features) → predictions
```

---

## 🟣 UNIT 9: HUGGING FACE

### Sentiment Classification con BERT

```python
from transformers import pipeline

# Pipeline simple
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

result = classifier("This movie is amazing!")
print(result)
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]

# Manual (más control)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "I love this!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
predicted_class = logits.argmax().item()

print(f"Predicted: {'POSITIVE' if predicted_class == 1 else 'NEGATIVE'}")
```

---

## 🟣 UNIT 9: RAG BÁSICO

### Retrieval Augmented Generation

```python
# Pseudocódigo (requiere ChromaDB, OpenAI, etc)

# 1. RETRIEVE: Busca documentos relevantes
db = ChromaDB("documents")
relevant_docs = db.search("¿Cuál es la capital de España?", top_k=3)

# 2. AUGMENT: Agrega al prompt
context = "\n".join([doc.text for doc in relevant_docs])
augmented_prompt = f"""
Basándote en estos documentos:
{context}

Responde la pregunta: ¿Cuál es la capital de España?
"""

# 3. GENERATE: LLM genera respuesta
llm = OpenAI("gpt-3.5-turbo")
response = llm.complete(augmented_prompt)
print(response)
# Output: "La capital de España es Madrid..."
```

---

## 📋 CHECKLIST PRÁCTICO PARA EXAMEN

### Antes de presentar cualquier solución:

```
REGRESIÓN:
☐ Cargar datos
☐ Train/test split
☐ Normalizar (fit train, transform ambos)
☐ Entrenar modelo
☐ Predecir
☐ Calcular MSE, RMSE, MAE, R²
☐ Hacer gráfica (real vs predicho)

CLASIFICACIÓN:
☐ Cargar datos
☐ Train/test split
☐ Normalizar
☐ Entrenar modelo
☐ Predecir
☐ Matriz de confusión
☐ Accuracy, Precision, Recall, F1
☐ Interpretation (cuál métrica es importante)

CLUSTERING:
☐ Cargar datos
☐ Elbow method (probar K=2 a K=10)
☐ Silhouette score para mejor K
☐ Entrenar K-Means con mejor K
☐ Ver centroides
☐ Visualizar clusters

DEEP LEARNING:
☐ Cargar datos (images/text)
☐ Dataloader
☐ Definir modelo (nn.Module)
☐ Loss (CrossEntropyLoss, MSE)
☐ Optimizer (Adam)
☐ Training loop (forward, backward, step)
☐ Evaluación (accuracy, loss)

TRANSFERENCIA:
☐ Cargar modelo preentrenado
☐ Congelar pesos iniciales
☐ Reemplazar última capa
☐ Entrenar solo última capa
☐ Evaluar
```

---

## 🎯 ERRORES COMUNES EN CÓDIGO

### ❌ Error 1: Data Leakage
```python
# MALO:
scaler.fit(X)  # Usa TODO
X_scaled = scaler.transform(X)
X_train, X_test = train_test_split(X_scaled, test_size=0.2)

# BIEN:
X_train, X_test = train_test_split(X, test_size=0.2)
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### ❌ Error 2: Predecir sin normalizar
```python
# MALO:
scaler.fit_transform(X_train)
model.fit(X_train, y_train)  # X_train not scaled!
model.predict(X_test)  # X_test not scaled!

# BIEN:
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model.fit(X_train_scaled, y_train)
model.predict(X_test_scaled)
```

### ❌ Error 3: Confundir fit() y fit_transform()
```python
# fit() retorna el objeto
# fit_transform() retorna datos transformados

scaler = StandardScaler()
scaler.fit(X_train)  # No retorna nada útil
X_train_scaled = scaler.transform(X_train)  # Transform después

# O en una línea:
X_train_scaled = scaler.fit_transform(X_train)
```

### ❌ Error 4: Random state inconsistente
```python
# MALO:
model1 = LogisticRegression(random_state=42)
model2 = LogisticRegression()  # random_state=None

# BIEN:
model1 = LogisticRegression(random_state=42)
model2 = LogisticRegression(random_state=42)
# Ambos reproducibles
```

### ❌ Error 5: No usar cross-validation
```python
# MALO:
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

# MEJOR:
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print(f"Media: {scores.mean()}, Std: {scores.std()}")
```

---

## 🚀 TIPS DE VELOCIDAD PARA EXAMEN

Si tienes POCO TIEMPO:

```python
# TEMPLATE MÁS RÁPIDO
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Carga
df = pd.read_csv('data.csv')
X, y = df.drop('target', axis=1), df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar (Random Forest es muy bueno, rápido)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

**Última actualización:** 12 Mayo 2026  
**Todo es copy-paste ready** ✅
