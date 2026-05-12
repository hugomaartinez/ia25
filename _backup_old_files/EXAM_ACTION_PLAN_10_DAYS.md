# 📅 PLAN DE ACCIÓN DEFINITIVO - 10 DÍAS (12-22 Mayo)

## 🎯 RESUMEN EJECUTIVO

**Hoy:** 12 de Mayo  
**Examen Teórico:** 22 de Mayo (10 días)  
**Examen Práctico:** 26 de Mayo (14 días)

**Estrategia:** Primero aprueba el teórico, luego enfócate en el práctico.

---

## 📚 ARCHIVOS CREADOS (LLEVA ESTOS AL EXAMEN)

### ⭐ IMPRESCINDIBLES (Llevar impresos o digitales)

1. **EXAM_QUICK_REFERENCE.md** (9.8 KB)
   - Código listo para copiar: pandas, numpy, sklearn
   - Workflow paso a paso para cualquier problema
   - Selección de modelo según tipo de problema
   - **USAR DURANTE:** Examen práctico

2. **EXAM_COMPLETE_SOLUTIONS.md** (27 KB)
   - 6 tipos de ejercicios resueltos (Regresión, Clasificación, Clustering, etc.)
   - 42 preguntas teóricas con respuestas detalladas
   - Trampas comunes y cómo evitarlas
   - **USAR DURANTE:** Ambos exámenes

3. **EXAM_DEEP_LEARNING_REFERENCE.md** (12 KB)
   - Código PyTorch listo para usar
   - Modelos: MLP, CNN, RNN, Transformers
   - Hugging Face examples
   - **USAR DURANTE:** Examen práctico (Deep Learning)

4. **EXAM_FULL_COURSE_COMPLETE.md** (27 KB)
   - 4 ejercicios Deep Learning resueltos con código
   - 25+ preguntas teóricas Units 6-10
   - Estrategia de estudio de 10 días
   - **USAR DURANTE:** Ambos exámenes

### 📖 COMPLEMENTARIOS (Para estudiar)

5. **EXAM_TEORICO_FULL_COURSE_STRATEGY.md** (9.6 KB)
   - Priorización por probabilidad (Qué entra más)
   - Tabla de conceptos clave por unit
   - Checklist final
   - **LEER:** Días 1-2

6. **EXAM_TEORICO_PREP_PLAN.md** (8.7 KB)
   - Conceptos clave memorización
   - Cronograma de 10 días detallado
   - Cuándo priorizar cada métrica
   - **LEER:** Días 1-2

7. **README_EXAM.md** (4.8 KB)
   - Estructura del examen
   - Tips durante el examen
   - Checklist pre-examen
   - **LEER:** Día 10

---

## 🔴 CRONOGRAMA AGRESIVO (10 DÍAS)

### **DÍAS 1-2 (12-13 Mayo) - REPASO RÁPIDO Units 1-3**

**Objetivo:** Recordar lo que ya sabes de la primer mitad

```
LEER (30 min):
  □ EXAM_TEORICO_FULL_COURSE_STRATEGY.md (priorización)
  □ EXAM_TEORICO_PREP_PLAN.md (conceptos críticos)

ESTUDIAR (1.5 horas):
  □ Matriz de confusión (TP, TN, FP, FN)
  □ Métricas: Accuracy, Precision, Recall, F1
  □ Regresión vs Clasificación
  □ Tres paradigmas: Supervisado, NoSupervisado, Refuerzo

HACER (1 hora):
  □ Lee EXAM_COMPLETE_SOLUTIONS.md Part 1 (primeras 3 preguntas)
  □ Intenta responder sin mirar las respuestas
  □ Revisa lo que falta
```

**TIEMPO TOTAL: 3 horas**

---

### **DÍAS 3-4 (14-15 Mayo) - Units 4 (Supervisado) + Métricas**

**Objetivo:** Dominar los algoritmos de clasificación y regresión

```
ESTUDIAR (2 horas):
  □ Linear Regression: y = β₀ + β₁X, minimiza MSE
  □ Logistic Regression: Sigmoide para probabilidades
  □ Decision Trees: max_depth es tu amigo (regularización)
  □ Random Forest: Bagging, votación, N estimators
  □ SVM: Hiperplano, margen, support vectors

CÓDIGO (1 hora):
  □ Abre EXAM_QUICK_REFERENCE.md
  □ Practica crear modelo → entrenar → evaluar
  □ Copia workflow en un notebook para practicar

TEÓRICA (1 hora):
  □ Lee preguntas 4-15 de EXAM_COMPLETE_SOLUTIONS.md
  □ Verifica respuestas
  □ Anota patrones de preguntas
```

**TIEMPO TOTAL: 4 horas**

---

### **DÍA 5 (16 Mayo) - Unit 5 (No Supervisado)**

**Objetivo:** Entender clustering y reducción de dimensionalidad

```
ESTUDIAR (1.5 horas):
  □ K-Means: Minimiza inercia, problema de inicialización
  □ DBSCAN: Densidad, epsilon, clusters irregulares
  □ PCA: Reducir dimensiones sin perder información
  □ Silhouette Score: Evaluar clustering (-1 a +1)

CÓDIGO (1 hora):
  □ Implementa K-Means simple
  □ Calcula Silhouette Score
  □ Práctica: Elbow method

TEÓRICA (30 min):
  □ Preguntas 16-25 de EXAM_COMPLETE_SOLUTIONS.md
  □ Entiende cuándo usar cada algoritmo
```

**TIEMPO TOTAL: 3 horas**

---

### **DÍAS 6-7 (17-18 Mayo) - Unit 6 (Deep Learning Basics)**

**Objetivo:** Entender redes neuronales desde cero

```
ESTUDIAR (1.5 horas):
  □ Perceptrón: Limitación, linealmente separable
  □ MLP: Múltiples capas, capacidad de aprendizaje
  □ Backpropagation: Flujo de gradientes (conceptualmente)
  □ Funciones activación: ReLU, Sigmoid, Tanh
  □ Loss functions: Cross-Entropy (clasificación), MSE (regresión)

CÓDIGO PyTorch (2 horas):
  □ Lee EXAM_DEEP_LEARNING_REFERENCE.md (primeras secciones)
  □ Sigue ejemplo simple MLP en MNIST
  □ Entiende: nn.Module, forward(), training loop
  □ Practica: Modifica el ejemplo para CIFAR-10

TEÓRICA (1 hora):
  □ Preguntas teóricas Unit 6 de EXAM_FULL_COURSE_COMPLETE.md
  □ Verifica entendimiento
```

**TIEMPO TOTAL: 4.5 horas**

---

### **DÍA 8 (19 Mayo) - Units 7-8 (CNNs + Transformers)**

**Objetivo:** Redes convolucionales y atención

```
ESTUDIAR (1.5 horas):
  □ CNN: Convoluciones, Pooling, Feature maps
  □ Transfer Learning: Usar modelos preentrenados
  □ RNNs/LSTMs: Memoria en secuencias
  □ Attention: Enfoque en partes relevantes
  □ Transformers: Self-attention, BERT vs GPT

CÓDIGO (1.5 horas):
  □ Implementa CNN simple en EXAM_DEEP_LEARNING_REFERENCE.md
  □ Transfer Learning: carga ResNet, modifica final layer
  □ Practica: Fine-tune en nuevo dataset

TEÓRICA (1 hora):
  □ Preguntas Units 7-8 de EXAM_FULL_COURSE_COMPLETE.md
```

**TIEMPO TOTAL: 4 horas**

---

### **DÍA 9 (20 Mayo) - Unit 9 (LLMs) + Simulacro**

**Objetivo:** Generative AI y preparación final

```
ESTUDIAR (1 hora):
  □ LLMs: Token prediction, generación de texto
  □ RAG: Retrieve + Augment + Generate
  □ Hugging Face: Cómo cargar modelos
  □ GANs: Generador vs Discriminador (concepto)
  □ Diffusion: Denoise iterativo

CÓDIGO (1 hora):
  □ Carga modelo preentrenado con Hugging Face
  □ Prueba sentimiento classification
  □ Entiende: tokenizer, model, inference

SIMULACRO (2 horas):
  □ Toma 20 preguntas aleatorias de tus archivos
  □ Responde en 30 minutos MÁXIMO
  □ Cronométrate
  □ Revisa errores y patrones

TEÓRICA (30 min):
  □ Preguntas Unit 9 de EXAM_FULL_COURSE_COMPLETE.md
```

**TIEMPO TOTAL: 4.5 horas**

---

### **DÍA 10 (21 Mayo) - REPASO FINAL + DESCANSO**

**Objetivo:** Confianza máxima, sin ansiedad

```
MAÑANA (2 horas):
  □ Revisa README_EXAM.md
  □ Repasa tu checklist de conceptos débiles
  □ Haz otro simulacro rápido (10 preguntas)
  □ Verifica formatos de archivos digitales

TARDE:
  □ Descansa, come bien, duerme 8 horas
  □ NO ESTUDIES DESPUÉS DE LAS 18:00
  □ Revisa que tienes los archivos listos (impresos o digitales)

IMPRESIONES (si tienes acceso a impresora):
  □ EXAM_COMPLETE_SOLUTIONS.md (para referencia)
  □ EXAM_QUICK_REFERENCE.md (código listo)
  □ Tus anotaciones personales
```

**TIEMPO TOTAL: 2 horas estudio + descanso**

---

## 📊 DISTRIBUCIÓN DE HORAS

```
Días 1-2:  3 horas  (Units 1-3: Fundamentos)
Días 3-4:  4 horas  (Unit 4: Supervisado) ← CRÍTICO
Día 5:     3 horas  (Unit 5: No supervisado) ← CRÍTICO
Días 6-7:  4.5 horas (Unit 6: Deep Learning)
Día 8:     4 horas  (Units 7-8: CNNs + Attention)
Día 9:     4.5 horas (Unit 9: LLMs + Simulacro)
Día 10:    2 horas  (Repaso + Descanso)

TOTAL: ~25 HORAS distribuidas en 10 días = 2.5 horas/día
```

---

## ✅ CHECKLIST PRE-EXAMEN (Día 22, mañana)

**6:00 AM:**
- [ ] He dormido 8 horas mínimo
- [ ] He desayunado bien
- [ ] Tengo laptop cargada

**7:00 AM - 30 min antes:**
- [ ] Tengo archivos impresos o digitales listos
- [ ] Llevo EXAM_QUICK_REFERENCE.md (código)
- [ ] Llevo EXAM_COMPLETE_SOLUTIONS.md (respuestas)
- [ ] Llevo EXAM_DEEP_LEARNING_REFERENCE.md (PyTorch)
- [ ] Llevo pen y papel para notas rápidas

**EN EL EXAMEN:**
- [ ] Leo cuidadosamente TODA la pregunta
- [ ] Identifico tipo: regresión, clasificación, clustering, DL
- [ ] Busco palabras clave en mis archivos
- [ ] Si no sé: elimino opciones obviamente falsas
- [ ] Gestiono tiempo: dedico menos a lo que domino, más a lo incierto

---

## 🚨 TRAMPAS A EVITAR

1. **Matriz de confusión:** Confundir filas/columnas
   - TP: predicción POSITIVA + real POSITIVA
   - FN: predicción NEGATIVA + real POSITIVA (peor error)

2. **Normalización:** Fit SOLO en train, transform en test
   - ❌ scaler.fit(X_test)
   - ✅ scaler.fit(X_train); scaler.transform(X_test)

3. **Data Leakage:** Calcular estadísticas DESPUÉS de split
   - ❌ mean = X.mean(); X = X - mean; entonces split
   - ✅ split primero; mean = X_train.mean()

4. **Overfitting:** Train error << Test error = problema
   - Solución: regularización, más datos, validación cruzada

5. **K-Means:** Sensible a inicialización
   - Siempre usa: `KMeans(n_init=10)` (múltiples intentos)

6. **Transfer Learning:** Congela pesos de capas iniciales
   - ❌ Actualizar todos los pesos
   - ✅ Freeze: `param.requires_grad = False`

---

## 💡 TIPS DURANTE EL TEST

### Si te toca pregunta de **Regresión:**
1. Identifica: Variables numéricas → output numérico
2. Métrica: MSE (penaliza grandes errores), MAE (robusto), R² (proporción)
3. Algoritmo: Linear, Polinomial, Ridge/Lasso

### Si te toca pregunta de **Clasificación:**
1. Crea matriz confusión (TP, TN, FP, FN)
2. Elige métrica según contexto:
   - Balanceadas: Accuracy
   - Desbalanceadas: F1, AUC-ROC
   - Medical: Recall (detectar todos los casos)
   - Spam: Precision (evitar falsos positivos)

### Si te toca pregunta de **Clustering:**
1. Problema: Agrupar SIN etiquetas
2. Métodos: K-Means, DBSCAN, Hierarchical
3. Evaluación: Silhouette Score, Elbow method

### Si te toca pregunta de **Deep Learning:**
1. Identifica datos: Imágenes → CNN, Texto → Transformer, Secuencias → RNN
2. Loss: Cross-Entropy (clasificación), MSE (regresión)
3. Pasos: Data → Model → Loss → Backward → Optimizer.step()

### Si te toca pregunta de **LLMs:**
1. RAG = Retrieve docs + Augment prompt + Generate respuesta
2. GPT = Autoregresivo (predice token siguiente)
3. BERT = Bidireccional (classification, no generación)

---

## 📱 VERSIÓN RÁPIDA (Último momento)

Si solo tienes 5 MINUTOS antes del examen:

**Lee esto:**
```
MATRIZ CONFUSIÓN:
TP, TN, FP, FN → Accuracy, Precision, Recall, F1

NORMALIZACIÓN:
fit(train), transform(train + test)

ALGORITMOS:
Regresión → Linear/Poly → MSE/RMSE/R²
Clasificación → Logistic/Tree/RF → Accuracy/Precision/Recall
Clustering → KMeans/DBSCAN → Silhouette Score

DEEP LEARNING:
Imágenes → CNN, Texto → Transformer
Loss = Cross-Entropy, Optimizer = Adam
Pasos: forward → loss → backward → step

LLMs:
RAG = Retrieve + Augment + Generate
GPT = Token prediction, BERT = Classification
```

---

## 🎯 ÚLTIMO RECORDATORIO

**Lo más importante no es memorizar TODO.**  
Es saber dónde buscar (en tus archivos) y aplicar lógica.

**Tienes estos archivos:**
- EXAM_COMPLETE_SOLUTIONS.md (42 preguntas teóricas resueltas)
- EXAM_QUICK_REFERENCE.md (código listo para copiar)
- EXAM_DEEP_LEARNING_REFERENCE.md (PyTorch ready-to-use)
- EXAM_FULL_COURSE_COMPLETE.md (ejercicios comple-

tos con soluciones)

**Durante el examen:**
1. Lee bien la pregunta
2. Busca en tus archivos
3. Aplica el código/concepto
4. Documenta tu lógica

---

**¡Tú puedes! 🚀**

Última actualización: 12 de Mayo 2026  
Examen Teórico: 22 de Mayo (10 días)  
Examen Práctico: 26 de Mayo (14 días)
