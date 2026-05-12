# 📖 TEORÍA - EXAMEN IA25 (Units 1-10)

**Fecha:** 12 Mayo 2026  
**Examen:** 22 Mayo 2026 (10 días)  
**Formato:** Test - Preguntas teóricas y conceptuales

---

## 🔴 UNITS 1-5: ML CLÁSICO (80% DEL EXAMEN)

### UNIT 1: FUNDAMENTOS

**¿Qué es AI/ML/DL?**
- **AI:** Máquinas que simulan inteligencia humana
- **ML:** Aprende de datos sin programación explícita
- **DL:** Redes neuronales con múltiples capas

**Paradigmas:**
- **Supervisado:** Con etiquetas (regresión, clasificación)
- **No supervisado:** Sin etiquetas (clustering, dimensionalidad)
- **Refuerzo:** Aprende con recompensas/castigos

**Strong AI vs Weak AI:**
- **Weak AI:** Resuelve tareas específicas (lo que existe hoy)
- **Strong AI:** Inteligencia general a nivel humano (futura)

---

### UNIT 2: ECOSYSTEM PYTHON

- **Python:** Lenguaje estándar para AI
- **NumPy:** Cálculos numéricos y arrays
- **Pandas:** Manipulación de datos (DataFrames)
- **Scikit-learn:** Modelos ML clásicos
- **Matplotlib/Seaborn:** Visualizaciones

---

### UNIT 3: PARADIGMAS BÁSICOS

**Supervisado:**
- **KNN (K-Nearest Neighbors):** Busca K vecinos más cercanos
  - Regresión: Promedia valores de K vecinos
  - Clasificación: Votación mayoritaria
  - Problema: Lento en predicción, sensible a escala

**No supervisado:**
- **K-Means:** Agrupa en K clusters minimizando inercia
  - Inercia = suma de distancias al centroide
  - Sensible a inicialización (usa n_init=10)
  - Necesita elegir K (Elbow method)

**Refuerzo:**
- **Q-Learning:** Aprende tabla Q (estado → acción)
  - Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
  - ε-Greedy: Balance entre exploración y explotación

---

### UNIT 4: SUPERVISED LEARNING (CRÍTICO)

#### **REGRESIÓN**

**Linear Regression:**
- Asume: y = β₀ + β₁X₁ + β₂X₂ + ...
- Minimiza: MSE = Σ(y - ŷ)² / n
- Interpretación: β=0.5 → X aumenta 1, y aumenta 0.5

**Polynomial Regression:**
- Grados mayores: y = β₀ + β₁X + β₂X² + ...
- Riesgo: Overfitting con grados muy altos

**Ridge/Lasso:**
- Ridge (L2): Penaliza coefficients grandes
- Lasso (L1): Reduce algunos coefficients a 0 (feature selection)

**Métricas Regresión:**

| Métrica | Fórmula | Qué mide | Cuándo usarla |
|---------|---------|----------|---|
| **MSE** | Σ(y-ŷ)²/n | Error promedio (penaliza errores grandes) | Estándar, pero sensible a outliers |
| **RMSE** | √MSE | Raíz de MSE (en misma unidad que y) | Interpretable (mismo rango que predicciones) |
| **MAE** | Σ\|y-ŷ\|/n | Error promedio (lineal) | Robusto a outliers, más realista que MSE |
| **R²** | 1-(SS_res/SS_tot) | % varianza explicada [0,1] | Evaluación global (qué tan bien predice) |

**Diferencias clave:**
- **MSE vs MAE:** MSE penaliza errores grandes más (x²), MAE los trata igual
  - Con outliers → MAE es mejor
  - Sin outliers → MSE es estándar
- **RMSE vs MSE:** RMSE en mismas unidades que y (más interpretable)
- **R²:** Mide qué porcentaje de varianza explica el modelo (0 = nada, 1 = perfecto)

#### **CLASIFICACIÓN**

**Logistic Regression:**
- Aplica sigmoide: σ(z) = 1/(1+e^-z) ∈ [0,1]
- Output: probabilidad de clase positiva
- Solo clasificación binaria (o one-vs-rest)

**Decision Trees:**
- Particiones recursivas del espacio
- **max_depth:** Regularizador (previene overfitting)
- Robusto a outliers (usa umbrales, no medias)
- Maneja categorías nativamente

**Random Forest:**
- Múltiples árboles con bootstrap (muestreo con reemplazo)
- Bagging: Reduce varianza mediante promediado
- Clasificación: Votación mayoritaria
- Regresión: Promedio de predicciones
- **n_estimators:** Cantidad de árboles (más = mejor pero lento)

**SVM (Support Vector Machine):**
- Encuentra hiperplano que maximiza margen
- Support vectors: Puntos fronterizos que definen el margen
- Kernel: Linear, RBF, Poly (no lineal)

#### **MATRIZ DE CONFUSIÓN Y MÉTRICAS**

```
              PREDICCIÓN
         +        -
       + TP      FN      (Real +)
REAL
       - FP      TN      (Real -)
```

**Memorizar (con ejemplos médicos):**
- **TP (True Positive):** Predicción SÍ, Real SÍ ✓
  - Modelo dice: "TIENES CÁNCER" → Realmente TIENES CÁNCER ✓
  - Acertó correctamente en caso positivo
  
- **TN (True Negative):** Predicción NO, Real NO ✓
  - Modelo dice: "NO tienes cáncer" → Realmente NO tienes ✓
  - Acertó correctamente en caso negativo
  
- **FP (False Positive):** Predicción SÍ, Real NO ❌
  - Modelo dice: "TIENES CÁNCER" → NO tienes cáncer ❌
  - **Error Tipo I:** Alarma falsa, susto innecesario
  - El modelo fue demasiado alarmista
  
- **FN (False Negative):** Predicción NO, Real SÍ ❌❌
  - Modelo dice: "NO tienes cáncer" → SÍ tienes cáncer ❌❌
  - **Error Tipo II - MÁS GRAVE:** Perdiste la oportunidad de tratamiento
  - El modelo fue demasiado relajista → PELIGROSO

**Derivadas (Métricas calculadas):**

```
Accuracy  = (TP+TN) / Total      → % Correcto GLOBAL (solo si clases balanceadas)
Precision = TP / (TP+FP)         → De lo que predije "+", ¿cuánto acerté realmente?
Recall    = TP / (TP+FN)         → De TODO lo que ES "+", ¿cuánto encontré?
F1        = 2×P×R / (P+R)        → Balance armónico de Precision y Recall
Specificity = TN / (TN+FP)       → De lo que ES "-", ¿cuánto rechacé correctamente?
Sensitivity = Recall             → Alias de Recall
```

**¿Qué mide cada una?**

| Métrica | Pregunta | Cuándo usarla |
|---------|----------|---|
| **Accuracy** | ¿Acerté en total? | Solo si clases bien balanceadas. CON desbalance es engañosa |
| **Precision** | Si digo "+", ¿de verdad lo es? | Cuando queremos EVITAR FALSOS POSITIVOS (Spam, Fraude) |
| **Recall** | ¿Encontré TODOS los casos "+"? | Cuando es CRÍTICO no perder ninguno (Medicina, Seguridad) |
| **F1** | ¿Balance Precision-Recall? | Cuando ambos importan igual (sin contexto específico) |
| **Specificity** | ¿Rechacé bien los "-"? | Raramente usada; Recall casi siempre es más importante |

**Regla de oro:**
- **Recall HIGH** → Minimiza FN (casos perdidos) → Medicina, detección crítica
- **Precision HIGH** → Minimiza FP (falsas alarmas) → Spam, bloqueos
- **F1** → Sin contexto claro, balance requerido
- **Accuracy** → SOLO si clases están equilibradas; con desbalance usa **AUC-ROC**

**Cuándo usar cada una:**
- **Precision Alto:** Spam (evitar falsos +), Fraude
- **Recall Alto:** Medicina (detectar TODOS los casos), Seguridad
- **F1:** Balance necesario
- **Accuracy:** Solo si clases bien balanceadas
- **AUC-ROC:** Mejor para clases desbalanceadas

---

### UNIT 5: UNSUPERVISED LEARNING

#### **K-Means**
- Minimiza inercia: WCSS = Σ distancia de punto a centroide
- Algoritmo: 1) Inicializa K centroides 2) Asigna puntos 3) Actualiza centroides 4) Repite
- **Problema:** Sensible a inicialización → **usa n_init=10**
- **Elbow method:** Gráfica inercia vs K, busca "codo"
- **n_clusters:** Hiperparámetro a elegir

#### **DBSCAN**
- Basado en densidad (agrupa puntos cercanos)
- Parámetros:
  - **eps:** Radio de búsqueda
  - **min_samples:** Mínimos puntos en eps para core point
- Ventaja: Detecta outliers, clusters irregulares
- Desventaja: 2 hiperparámetros a tunar

#### **PCA (Principal Component Analysis)**
- Reducción de dimensionalidad
- Encuentra ejes de máxima varianza
- Componentes principales ortogonales
- **n_components:** Cuántas dimensiones mantener
- Pierde interpretabilidad pero reduce complejidad

#### **Silhouette Score**
```
Rango: -1 a +1
+1 = Clusters densos y bien separados (perfecto)
 0 = Clusters solapados (mediocre)
-1 = Asignación errónea (malo)
```

#### **Anomaly Detection**
- Encuentra outliers (puntos anormales)
- Métodos: Isolation Forest, LOF, Z-score
- Uso: Fraude, defectos de manufactura

---

## 🟠 UNITS 6-10: DEEP LEARNING (15-20% DEL EXAMEN)

### UNIT 6: DEEP LEARNING FUNDAMENTALS

#### **Perceptrón**
- Unidad básica: suma ponderada + activación
- Formula: y = σ(w·x + b)
- **Limitación:** Solo linealmente separable
- Necesita MLP para problemas no lineales

#### **MLP (Multi-Layer Perceptron)**
- Múltiples capas: Input → Ocultas → Output
- Cada capa agrega no-linealidad
- Capacidad de aprender cualquier función

#### **Backpropagation**
- Propaga error hacia atrás
- Calcula gradientes layer por layer
- Usa chain rule de cálculo
- **Concepto clave:** Flujo de gradientes de output → input

#### **Funciones de Activación**

```
ReLU(x)     = max(0, x)
Sigmoid(x)  = 1/(1+e^-x)
Tanh(x)     = (e^x-e^-x)/(e^x+e^-x)
Softmax     = e^xi / Σe^xj
```

**Idea simple:** una activación decide cómo se transforma la suma de entrada de una neurona antes de pasar a la siguiente capa.

**Intuición de cada una:**
- **ReLU:** deja pasar los valores positivos y bloquea los negativos. Es como un filtro que dice: "si no aporta, lo apago".
- **Sigmoid:** comprime cualquier número entre 0 y 1. Se interpreta fácil como probabilidad.
- **Tanh:** comprime entre -1 y 1. Se parece a sigmoid, pero centrada en 0, así que puede dar valores positivos y negativos.
- **Softmax:** convierte varias salidas en probabilidades que suman 1. Sirve cuando solo puede haber una clase correcta.

**Cuándo usar:**
- **Hidden layers:** ReLU, porque suele entrenar mejor y es la opción estándar.
- **Binary classification output:** Sigmoid, porque devuelve una probabilidad de la clase positiva.
- **Multi-class output:** Softmax, porque reparte la probabilidad entre todas las clases.
- **Regression output:** salida lineal, sin activación, porque queremos cualquier valor real.

**Regla rápida para memorizar:**
- ReLU = "dejo pasar lo positivo"
- Sigmoid = "lo convierto en probabilidad de 0 a 1"
- Tanh = "como sigmoid, pero de -1 a 1"
- Softmax = "elijo una clase entre varias"

#### **Loss Functions**

```
Cross-Entropy = -Σ y·log(ŷ)
MSE          = Σ(y-ŷ)²/n
Binary Cross-Entropy = -(y·log(ŷ) + (1-y)·log(1-ŷ))
```

**Idea simple:** una loss mide cuán mal lo está haciendo el modelo. Cuanto más pequeña, mejor.

**Intuición de cada una:**
- **MSE:** compara valor real y valor predicho y castiga mucho los errores grandes. Se usa en regresión porque el objetivo es predecir un número.
- **Cross-Entropy:** mide si el modelo le dio mucha probabilidad a la clase correcta. Se usa en clasificación porque no queremos solo acertar la clase, sino asignarle alta confianza.
- **Binary Cross-Entropy:** es la versión de cross-entropy para dos clases. Se usa cuando la salida es 0/1 o negativo/positivo.

**Cuándo usar:**
- **Regresión:** MSE.
- **Clasificación binaria:** Binary Cross-Entropy.
- **Clasificación multiclase:** Cross-Entropy.

**Regla rápida para memorizar:**
- Si predigo un número → **MSE**
- Si predigo sí/no → **Binary Cross-Entropy**
- Si predigo una clase entre varias → **Cross-Entropy**

---

### UNIT 7: CONVOLUTIONAL NEURAL NETWORKS (CNNs)

#### **Conceptos**

**Convolución:**
- Desliza filtro (kernel) sobre imagen
- Aprende features locales (bordes, esquinas, texturas)
- Output: Feature map

**Pooling:**
- Reduce dimensionalidad
- Max Pooling: Toma valor máximo
- Average Pooling: Promedia
- Mantiene features importantes

**Stride y Padding:**
- Stride: Cuánto se mueve el kernel
- Padding: Añade bordes (0s) para mantener tamaño

#### **Arquitecturas Comunes**

- **LeNet:** Primeras CNNs (MNIST)
- **AlexNet:** Breakthrough en ImageNet 2012
- **VGG:** Bloques simples repetidos
- **ResNet:** Conexiones residuales (skip connections)
- **Inception/GoogleNet:** Convoluciones en paralelo

#### **Transfer Learning**
- Carga modelo preentrenado (ImageNet)
- Congela pesos iniciales (ya aprendió features)
- Reemplaza última capa con nuevo clasificador
- Fine-tune o solo entrenar última capa
- **Ventaja:** Menos datos, entrenamiento rápido

---

### UNIT 8: TRANSFORMERS & RECURRENT NETWORKS

#### **RNNs**
- Recurrencia: Estado oculto h(t) usa h(t-1)
- Secuencias: Procesa tokens uno por uno
- Problema: Vanishing gradient (información se pierde)

#### **LSTMs (Long Short-Term Memory)**
- Gates (puertas): Forget, Input, Output
- Mantiene cell state (memoria a largo plazo)
- Resuelve vanishing gradient
- Mejor para secuencias largas

#### **Attention Mechanism**
- "¿A qué partes del input debo enfocarme?"
- Scores: Importancia de cada posición
- Calcula: Queries (Q), Keys (K), Values (V)
- **Formula:** Attention(Q,K,V) = softmax(Q·K^T/√d)·V

#### **Transformers**
- **Self-Attention:** Cada token ve todos los demás
- **Multi-Head Attention:** Múltiples perspectivas simultáneamente
- **Positional Encoding:** Inyecta información de orden/posición
- **Feed-Forward:** Capas densas por posición
- **NO usa recurrencia:** Procesa en paralelo (rápido)

#### **BERT vs GPT**

**BERT:**
- Bidireccional: Ve contexto izquierda Y derecha
- Pre-training: Masked Language Model
- Uso: Clasificación, NER, similar
- NO es generativo

**GPT:**
- Autoregresivo: Predice siguiente token (izquierda a derecha)
- Pre-training: Language Modeling
- Uso: Generación de texto, Q&A, chat
- MÁS generativo

---

### UNIT 9: GENERATIVE AI & LLMs

#### **LLMs (Large Language Models)**
- Redes neuronales masivas (miles de millones de parámetros)
- Predicción de siguiente token dado contexto anterior
- Training: Predecir token siguiente en corpus gigante
- Inference: Token por token (autoregresivo)

#### **Token**
- Unidad mínima: palabra o sub-palabra
- "Hello world" = 2-3 tokens
- Los LLMs trabajan con tokens, no caracteres

#### **Prompt Engineering**
- Cómo pedir al LLM correctamente
- Context setting, ejemplos (few-shot), clarity
- Affect output quality significativamente

#### **RAG (Retrieval Augmented Generation)**
1. **Retrieve:** Busca documentos relevantes en base de datos
2. **Augment:** Agrega documentos al prompt original
3. **Generate:** LLM genera respuesta CON contexto

**Ventaja:** LLM accede a tus datos sin reentrenarse

#### **GANs (Generative Adversarial Networks)**
- **Generador:** Crea datos falsos
- **Discriminador:** Detecta si son falsos
- Compiten → ambos mejoran
- Muy efectivo para generación de imágenes

#### **Diffusion Models**
- Iterativamente remueve ruido de imagen
- Training: Aprende a remover ruido
- Inference: Añade ruido, luego remueve iterativamente
- SOTA en generación de imágenes (DALL-E, Stable Diffusion)

#### **Hugging Face**
- Librería para usar modelos preentrenados
- Models, Datasets, Tokenizers
- Fácil acceso a BERT, GPT, etc.

---

### UNIT 10: MLOps (MENOS PROBABLE)

#### **Deployment**
- **Batch:** Predicciones en lotes (cada hora/día)
- **Real-time:** API que predice por request
- **Edge:** Modelo en dispositivo (móvil)

#### **Monitoring**
- Vigilar rendimiento del modelo en producción
- Data drift: Datos cambians con tiempo
- Concept drift: Relación entre features y target cambia
- Retraining cuando rendimiento baja

#### **CI/CD**
- Continuous Integration: Tests automáticos
- Continuous Deployment: Deploy automático
- Previene bugs en producción

#### **Docker**
- Containeriza código + dependencias
- Reproducible en cualquier máquina
- Estándar en MLOps

#### **Model Versioning**
- Guardar modelos y metadata
- MLflow, Weights & Biases
- Trackear qué modelo está en producción

---

## 🎯 CONCEPTOS CLAVE POR UNIT (MEMORIZAR)

### Unit 1: AI, ML, DL, Paradigmas
### Unit 2: Python, NumPy, Pandas, Sklearn
### Unit 3: KNN, K-Means, Q-Learning
### Unit 4: Regresión, Clasificación, Metrics, Matriz Confusión ⭐⭐⭐
### Unit 5: Clustering, PCA, Silhouette
### Unit 6: Perceptrón, MLP, Backprop, Activaciones, Loss
### Unit 7: Convolution, Pooling, Transfer Learning
### Unit 8: RNN, LSTM, Attention, Transformer, BERT vs GPT
### Unit 9: LLM, RAG, GANs, Diffusion, Hugging Face
### Unit 10: Deployment, Monitoring, Docker, MLOps

---

## 📋 PREGUNTAS TIPO TEST (BASADAS EN EXAMEN ANTERIOR)

### SECCIÓN 1: MÉTRICAS (12 preguntas)

**P1:** ¿Cuál es la matriz de confusión?
**R:** TP, TN, FP, FN - representan predicciones correctas/incorrectas

**P2:** ¿Cuándo usar Precision vs Recall?
**R:** Precision = evitar FP (spam); Recall = detectar todos (medicina)

**P3:** ¿Qué es accuracy y cuándo es engañosa?
**R:** (TP+TN)/Total; engañosa con clases desbalanceadas

**P4:** ¿F1 qué mide?
**R:** Balance entre Precision y Recall

**P5:** ¿R² qué significa?
**R:** Proporción de varianza explicada (0 a 1)

**P6:** ¿MSE vs MAE?
**R:** MSE penaliza grandes errores; MAE robusto a outliers

**P7:** ¿Silhouette Score rango?
**R:** -1 a +1 (1=perfecto, 0=solapado, -1=malo)

**P8:** ¿Cuándo usar RMSE?
**R:** Cuando errores grandes son inaceptables (misma unidad que y)

**P9:** ¿Specificity qué mide?
**R:** TN/(TN+FP) - % verdaderos negativos detectados

**P10:** ¿AUC-ROC para qué?
**R:** Evaluar clases desbalanceadas (mejor que Accuracy)

**P11:** ¿Accuracy con clases balanceadas?
**R:** Métrica válida

**P12:** ¿Cross-validation para qué?
**R:** Evaluar más confiable dividiendo datos en K partes

---

### SECCIÓN 2: REGRESIÓN (8 preguntas)

**P1:** Linear Regression asume:
**R:** y = β₀ + β₁X (relación lineal)

**P2:** ¿Qué minimiza Linear Regression?
**R:** MSE

**P3:** Coeficiente β=0.5 significa:
**R:** Por cada unidad de X, y aumenta 0.5

**P4:** Polynomial Regression problema:
**R:** Overfitting con grados muy altos

**P5:** Ridge vs Lasso:
**R:** Ridge penaliza coefficients (L2); Lasso reduce algunos a 0 (L1)

**P6:** ¿Cuándo usar Poly Regression?
**R:** Cuando relación es claramente no-lineal

**P7:** Underfitting en regresión:
**R:** Modelo simple, train error alto, test error alto (error similar)

**P8:** Overfitting en regresión:
**R:** Modelo complejo, train error bajo, test error alto (train << test)

---

### SECCIÓN 3: CLASIFICACIÓN (10 preguntas)

**P1:** Logistic Regression para:
**R:** Clasificación binaria con probabilidades [0,1]

**P2:** Sigmoide rango:
**R:** 0 a 1

**P3:** Decision Tree max_depth:
**R:** Regularizador que previene overfitting

**P4:** Random Forest bagging:
**R:** Muestreo con reemplazo, reduce varianza mediante promediado

**P5:** Clasificación en Random Forest:
**R:** Votación mayoritaria de árboles

**P6:** SVM qué busca:
**R:** Hiperplano que maximiza margen

**P7:** KNN clasificación:
**R:** Votación mayoritaria de K vecinos

**P8:** KNN problema:
**R:** Lento en predicción, sensible a escala

**P9:** Decision Tree robusto a:
**R:** Outliers (usa umbrales, no medias)

**P10:** Cuando árboles overfitean:
**R:** max_depth muy alto

---

### SECCIÓN 4: CLUSTERING (6 preguntas)

**P1:** K-Means minimiza:
**R:** Inercia (distancia al centroide)

**P2:** Problema K-Means:
**R:** Sensible a inicialización (usar n_init=10)

**P3:** Elbow method para:
**R:** Elegir número óptimo de clusters K

**P4:** DBSCAN para:
**R:** Clusters irregulares, detecta outliers

**P5:** PCA para:
**R:** Reducción de dimensionalidad

**P6:** Silhouette negativo:
**R:** Punto asignado al cluster incorrecto

---

### SECCIÓN 5: DATOS (8 preguntas)

**P1:** MCAR vs MAR:
**R:** MCAR=aleatoria pura (borrar OK); MAR=en variables observadas (imputar)

**P2:** MNAR:
**R:** Faltante en valor propio (difícil, riesgo sesgo)

**P3:** Normalización StandardScaler:
**R:** x' = (x - mean) / std (rango indefinido)

**P4:** MinMaxScaler:
**R:** x' = (x - min) / (max - min) (rango [0,1])

**P5:** MinMaxScaler problema:
**R:** Muy sensible a outliers (estira rango)

**P6:** Data Leakage:
**R:** fit scaler en TODO, test set afecta train

**P7:** Correcta normalización:
**R:** fit(train), transform(train+test)

**P8:** Listwise deletion:
**R:** Borra fila si falta un valor (pierde información)

---

### SECCIÓN 6: OVERFITTING (6 preguntas)

**P1:** Overfitting síntoma:
**R:** train error << test error

**P2:** Underfitting síntoma:
**R:** train error ≈ test error (ambos altos)

**P3:** Solución overfitting:
**R:** Menos parámetros, regularización, más datos, validación cruzada

**P4:** Curva aprendizaje overfitting:
**R:** train baja, val sube

**P5:** Train/Test split ratio:
**R:** 70% train, 15% val, 15% test

**P6:** Validación cruzada K-fold:
**R:** Divide train en K, entrena K veces

---

### SECCIÓN 7: PARADIGMAS (5 preguntas)

**P1:** Supervisado:
**R:** Con etiquetas (regresión, clasificación)

**P2:** No supervisado:
**R:** Sin etiquetas (clustering, dimensionalidad)

**P3:** Refuerzo:
**R:** Aprende interactuando, recompensas/castigos

**P4:** Multietiqueta:
**R:** Múltiples etiquetas simultáneas

**P5:** Semi-supervisado:
**R:** Pocas etiquetas, mucho no-etiquetado

---

### SECCIÓN 8: DEEP LEARNING (8 preguntas)

**P1:** Perceptrón limitación:
**R:** Solo linealmente separable

**P2:** MLP para:
**R:** Problemas no-lineales (múltiples capas)

**P3:** Backpropagation:
**R:** Propaga error atrás, calcula gradientes

**P4:** ReLU:
**R:** max(0, x), estándar en capas ocultas

**P5:** Sigmoid:
**R:** 1/(1+e^-x), output probabilidades [0,1]

**P6:** Cross-Entropy:
**R:** -Σy·log(ŷ), loss para clasificación

**P7:** CNN para:
**R:** Imágenes (convoluciones capturan features locales)

**P8:** Transfer Learning:
**R:** Carga preentrenado, congela pesos, entrena última capa

---

### SECCIÓN 9: TRANSFORMERS (5 preguntas)

**P1:** Attention qué hace:
**R:** Pondera importancia de cada posición

**P2:** Self-Attention:
**R:** Token ve todos los otros tokens

**P3:** Multi-Head:
**R:** Múltiples perspectives simultáneamente

**P4:** BERT para:
**R:** Clasificación, NER (bidireccional)

**P5:** GPT para:
**R:** Generación de texto (autoregresivo)

---

### SECCIÓN 10: LLMs (4 preguntas)

**P1:** LLM qué predice:
**R:** Token siguiente dado contexto anterior

**P2:** RAG pasos:
**R:** Retrieve docs → Augment prompt → Generate respuesta

**P3:** GANs:
**R:** Generador vs Discriminador compiten

**P4:** Diffusion Models:
**R:** Remueve ruido iterativamente

---

## 🎯 TIPS PARA PASAR EL TEST

### Estrategia General:
1. **Lee TODO la pregunta** (no confundas TP con TN)
2. **Identifica tema:** ¿Métrica, Algoritmo, Overfitting?
3. **Busca en cheat sheet** si es necesario (fórmulas)
4. **Elimina opciones obviamente falsas**
5. **Si tienes 50% seguridad, contesta** (mejor que nada)

### Por tipo de pregunta:

**Si pregunta sobre MÉTRICA:**
- Piensa: ¿Qué error me importa más?
- Precision → evitar FP
- Recall → detectar todos
- F1 → balance

**Si pregunta sobre ALGORITMO:**
- ¿Qué hace? (Regresión, Clasificación, Clustering)
- ¿Cuándo se usa?
- ¿Cuál es el hiperparámetro clave?

**Si pregunta sobre OVERFITTING:**
- train << test → overfitting
- train ≈ test (ambos altos) → underfitting
- Soluciones: regularización, más datos

**Si pregunta sobre CONCEPTOS:**
- ReLU = max(0,x) en capas ocultas
- Sigmoid = probabilidades [0,1] en output
- Cross-Entropy = clasificación
- MSE = regresión

---

## 🔴 TRAMPAS FRECUENTES

1. **Confundir TP/FP:** TP=ambos+, FP=predigo+ pero real-
2. **Accuracy con desbalance:** Engañosa, usar F1 o AUC
3. **Data Leakage:** Calcular mean ANTES de split
4. **MinMaxScaler outliers:** Estira el rango
5. **K-Means sin n_init:** Puede quedar en mínimo local
6. **Overfitting vs Underfitting:** train<<test es OVER, train≈test es UNDER
7. **Precision vs Recall:** No son lo mismo, contexto determina

---

**Última actualización:** 12 Mayo 2026  
**Objetivo:** 70%+ en examen teórico

¡Memoriza esto y pasas! 💪
