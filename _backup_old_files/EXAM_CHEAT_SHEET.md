# 📋 CHEAT SHEET - VERSIÓN MÓVIL (Copiar en notas del celular)

## MATRIZ DE CONFUSIÓN (BASE DE TODO)

```
              PREDICCIÓN
         +        -
       +  TP      FN
REAL
       -  FP      TN
```

**Memorizar:**
- TP: Ambos positivos ✓
- TN: Ambos negativos ✓
- FP: Predicción SÍ, Real NO (Email importante → spam)
- FN: Predicción NO, Real SÍ (Email spam → importante - PEOR)

---

## MÉTRICAS (QUÉ CALCULAN)

### Clasificación
```
Accuracy  = (TP+TN) / Total      → % correcto (puede engañar)
Precision = TP / (TP+FP)         → "Confiabilidad de predicción +"
Recall    = TP / (TP+FN)         → "Detección de todos los +"
F1        = 2 × (P × R) / (P+R)  → Balance de P y R
AUC-ROC   = Curva ROC            → Mejor para desbalanceadas
```

**Cuándo usar:**
- Precision: Spam (no quiero falsos +)
- Recall: Medicina (quiero detectar TODOS los casos)
- F1: Balance necesario
- Accuracy: Solo si clases balanceadas

### Regresión
```
MSE   = Σ(y - ŷ)² / n           → Penaliza grandes errores
RMSE  = √MSE                     → Misma unidad que y
MAE   = Σ|y - ŷ| / n            → Robusto a outliers
R²    = 1 - (SS_res / SS_tot)    → % varianza explicada (0-1)
```

**Cuándo usar:**
- RMSE: Errores grandes → inaceptables
- MAE: Hay outliers
- R²: Siempre, para interpretar

### Clustering
```
Silhouette Score: -1 a +1
  +1 = Perfecto
   0 = Solapado
  -1 = Mal
```

---

## ALGORITMOS CLÁSICOS (UNITS 1-5)

### Supervisado - Regresión
| Algoritmo | Función | Parámetro |
|-----------|---------|-----------|
| Linear Reg | Línea recta | Coef |
| Poly Reg | Polinomio | degree |
| Ridge | + Regularización L2 | alpha |
| Lasso | + Regularización L1 | alpha |

### Supervisado - Clasificación
| Algoritmo | Función | Parámetro |
|-----------|---------|-----------|
| Logistic | Sigmoide [0,1] | C |
| Decision Tree | Particiones | max_depth |
| Random Forest | Múltiples árboles | n_estimators |
| SVM | Hiperplano máximo | C, kernel |
| KNN | K vecinos cercanos | n_neighbors |

### No Supervisado
| Algoritmo | Función | Parámetro |
|-----------|---------|-----------|
| K-Means | Agrupación | n_clusters, n_init |
| DBSCAN | Densidad | eps, min_samples |
| PCA | Reducir dimensiones | n_components |

---

## WORKFLOW ESTÁNDAR

```
1. CARGAR DATOS
   import pandas as pd
   df = pd.read_csv('data.csv')

2. EXPLORACIÓN (EDA)
   df.head(), df.info(), df.describe()
   df.isnull().sum()

3. DATOS FALTANTES
   Si MCAR: borrar OK
   Si MAR/MNAR: imputar (media, mediana, KNN)

4. VARIABLES CATEGÓRICAS
   pd.get_dummies(df, columns=['cat_col'])

5. NORMALIZACIÓN (fit TRAIN, transform TRAIN+TEST)
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
   X_test = scaler.transform(X_test)  # NO fit

6. SPLIT (70% train, 15% val, 15% test)
   from sklearn.model_selection import train_test_split
   X_train, X_test = train_test_split(X, 0.8)
   X_train, X_val = train_test_split(X_train, 0.5)

7. MODELO
   model = LogisticRegression()
   model.fit(X_train, y_train)

8. PREDICCIÓN
   y_pred = model.predict(X_test)

9. EVALUACIÓN
   from sklearn.metrics import accuracy_score, confusion_matrix
   acc = accuracy_score(y_test, y_pred)
```

---

## DEEP LEARNING BASICS (UNIT 6)

### PyTorch Workflow
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training Loop
for epoch in range(epochs):
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
```

### Funciones Activación
```
ReLU(x)    = max(0, x)          → Capas ocultas (estándar)
Sigmoid(x) = 1/(1+e^-x)         → Output clasificación binaria [0,1]
Tanh(x)    = (e^x - e^-x)/(e^x + e^-x)  → Rango [-1, 1]
```

### Loss Functions
```
CrossEntropyLoss → Clasificación (+ softmax)
MSE              → Regresión
```

---

## CNNs (UNIT 7)

### Componentes
```
Conv2d(in_channels, out_channels, kernel_size=3)
    → Aplicar filtro a imagen

MaxPool2d(2)
    → Bajar resolución, mantener features importantes

Flatten()
    → Convertir a vector para capas densas

Linear()
    → Clasificación final
```

### Arquitectura Simple
```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64*5*5, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # (N, 32, 31, 31)
        x = self.pool(torch.relu(self.conv2(x)))  # (N, 64, 14, 14)
        x = x.view(x.size(0), -1)                  # Flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

---

## TRANSFORMERS (UNIT 8)

### Conceptos (NO necesitas fórmulas)
```
Attention      → "¿A qué partes del input debo enfocarse?"
Self-Attention → Cada posición ve todas las otras
Multi-Head     → Múltiples "perspectivas" simultáneamente
Transformer    → Attention + Capas densas (sin RNN)
Positional Encoding → Inyecta información de orden
```

### Modelos Preentrenados
```
BERT       → Bidireccional, para CLASSIFICATION
GPT        → Autoregresivo, para GENERACIÓN
Hugging Face → Librería para usar estos modelos
```

### Código Hugging Face
```python
from transformers import pipeline

# Classification
pipe = pipeline("sentiment-analysis")
result = pipe("This movie is amazing!")

# Generation
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Manual
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base")
tokens = tokenizer("Hello world", return_tensors="pt")
```

---

## LLMs & RAG (UNIT 9)

### Conceptos
```
LLM               → Red neuronal masiva que predice el siguiente token
Token             → Palabra o carácter (ej: "Hello" = 1 token)
Prompt Engineering → Cómo pedir al LLM correctamente
```

### RAG (Retrieval Augmented Generation)
```
1. RETRIEVE  → Busca documentos relevantes en base de datos
2. AUGMENT   → Agrega documentos al prompt original
3. GENERATE  → LLM genera respuesta con contexto

Ventaja: LLM usa tus datos sin reentrenarse
```

### Generative Models
```
GAN (Generative Adversarial Network):
  - Generador: Crea datos falsos
  - Discriminador: Detecta falsos
  - Compiten hasta convergencia

Diffusion Models:
  - Añade ruido iterativamente
  - Aprende a remover ruido
  - Genera muestras paso a paso
```

---

## ERRORES COMUNES (TRAMPAS)

### ❌ Data Leakage
```
MALO:
mean = X.mean()  # Calcula en TODO el dataset
X = X - mean
X_train, X_test = train_test_split(X, 0.8)

BIEN:
X_train, X_test = train_test_split(X, 0.8)
mean = X_train.mean()  # SOLO train
X_train = X_train - mean
X_test = X_test - mean
```

### ❌ Normalización
```
MALO:
scaler.fit(X_train)
X_train = scaler.transform(X_train)
scaler.fit(X_test)      # ❌❌❌
X_test = scaler.transform(X_test)

BIEN:
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Sin fit
```

### ❌ Confundir Precision vs Recall
```
Precision = TP / (TP+FP) 
  → "De lo que predije +, cuántos eran realmente +"
  → Importa cuando FP caro (spam, fraude)

Recall = TP / (TP+FN)
  → "De todos los + reales, cuántos detecté"
  → Importa cuando FN caro (medicina, seguridad)
```

### ❌ K-Means sin n_init
```
MALO:
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)  # Puede converger a mínimo local

BIEN:
kmeans = KMeans(n_clusters=3, n_init=10)  # Múltiples intentos
kmeans.fit(X)
```

---

## PREGUNTAS TIPO TEST

### Si pregunta sobre **Métrica de evaluación:**
→ Piensa en: ¿Qué errores me importan más?
→ FP: Precision ↑
→ FN: Recall ↑

### Si pregunta sobre **Tipo de problema:**
→ ¿Con etiquetas? Supervisado
→ ¿Sin etiquetas? No supervisado
→ ¿Agrupación? Clustering
→ ¿Output numérico? Regresión
→ ¿Output categoría? Clasificación

### Si pregunta sobre **Algoritmo:**
→ ¿Tabular? sklearn (Tree, RF, KNN, Logistic)
→ ¿Imágenes? CNN
→ ¿Texto? Transformer / Hugging Face
→ ¿Secuencias? RNN / LSTM

### Si pregunta sobre **Overfitting:**
→ Train error << Test error = Memorización
→ Solución: Más datos, regularización, validación cruzada, reducir parámetros

---

## DURANTE EL EXAMEN

1. **Lee TODO la pregunta** (A veces tienen trampa)
2. **Identifica tipo** (Regresión, clasificación, clustering, DL)
3. **Busca en tus archivos** (Código, conceptos, ejemplos)
4. **Aplica lógica** (Adapta al problema específico)
5. **Documenta** (Explica qué haces y por qué)

---

**Última actualización:** 12 Mayo 2026  
**¡Buena suerte! 🚀**
