# Estrategia Examen Teórico - CURSO COMPLETO (Units 1-10)

## ⚠️ REALIDAD

**Tiempo disponible:** 10 días  
**Contenido:** Units 1-10 completas  
**Formato:** Test tipo (probablemente 100-150 preguntas)

**Conclusión:** Es IMPOSIBLE dominar todo. Necesitamos ESTRATEGIA.

---

## 📊 PRIORIZACIÓN POR PROBABILIDAD DE ENTRAR

### 🔴 GARANTIZADO ENTRA (80% del examen)

**Units 1-5: ML Clásico** ← Lo que ya estudiaste en primer parcial
- Métricas de evaluación
- Regresión
- Clasificación  
- Clustering
- Manejo de datos
- Paradigmas de ML

**Esto YA lo viste.** Solo necesitas repasar.

### 🟠 PROBABLEMENTE ENTRA (15% del examen)

**Unit 6: Deep Learning Fundamentos**
- Perceptrón, MLP
- Backpropagation (concepto, no fórmula)
- Funciones de activación (ReLU, sigmoid)
- PyTorch basics

**Unit 9: Generative AI & LLMs**
- Concepto de LLMs
- RAG (Retrieval Augmented Generation)
- Transformers (concepto alto nivel)

### 🟡 PUEDE ENTRAR (5% del examen)

**Unit 7: CNNs** - Si hay preguntas conceptuales sobre convoluciones  
**Unit 8: Transformers** - Si hay preguntas sobre attention  
**Unit 10: MLOps** - Muy poco probable en test teórico  

---

## 🎯 PLAN REALISTA (10 DÍAS)

### Distribución recomendada:

```
Días 1-4:   REPASO Units 1-5 (80% del examen)
            ├─ Métricas (2 días)
            ├─ Regresión/Clasificación (1 día)
            └─ Datos, Clustering, Paradigmas (1 día)

Días 5-8:   Deep Learning Basics (15% del examen)
            ├─ Conceptos: Perceptrón, Backprop
            ├─ Activaciones, Loss functions
            ├─ PyTorch basics
            └─ CNNs conceptualmente

Días 9-10:  LLMs & Transformers (5% del examen)
            ├─ Qué es un LLM
            ├─ Transformers (alto nivel)
            ├─ RAG
            └─ REPASO GENERAL
```

---

## 📝 GUÍA RÁPIDA - QUÉ ESTUDIAR DE CADA UNIT

### ✅ UNIT 1: Fundamentos (5 preguntas esperadas)
**ESTUDIA:**
- ¿Qué es AI? ¿Qué es ML? ¿Qué es DL?
- Strong AI vs Weak AI
- Paradigmas: Supervisado, No supervisado, Refuerzo

**NO NECESITAS:** Historia detallada, filosofía

---

### ✅ UNIT 2: Programming Ecosystem (2 preguntas)
**ESTUDIA:**
- Python es estándar para AI
- NumPy para cálculos
- Pandas para manipulación

**NO NECESITAS:** Detalles de Git, setup

---

### ✅ UNIT 3: Tres Paradigmas (5 preguntas)
**ESTUDIA:**
- KNN (Supervisado)
- K-Means (No supervisado)
- Concepto de Q-Learning (Refuerzo)

**NO NECESITAS:** Implementación detallada

---

### ✅ UNIT 4: Supervised Learning PROFUNDO (20 preguntas)
**ESTUDIA (CRÍTICO):**
- Linear Regression: y = β₀ + β₁X, minimiza MSE
- Logistic Regression: Sigmoide para probabilidades
- Decision Trees: Particiones, max_depth
- Random Forest: Bagging, votación/promedio
- SVM: Hiperplano, margen
- Evaluación: Todas las métricas

**NO NECESITAS:** Derivaciones matemáticas

---

### ✅ UNIT 5: Unsupervised Learning (12 preguntas)
**ESTUDIA:**
- K-Means: Minimiza inercia, n_init
- DBSCAN: Densidad, ε
- PCA: Reducción dimensionalidad
- Anomaly Detection: Outliers

**NO NECESITAS:** Matemática de PCA

---

### ✅ UNIT 6: Deep Learning Fundamentals (8 preguntas)
**ESTUDIA:**
- Perceptrón: Limitación linealmente separables
- MLP: Múltiples capas
- Backpropagation: Flujo de gradientes (concepto)
- Funciones de activación:
  - ReLU: max(0, x) → No linealidad
  - Sigmoid: Probabilidades (0,1)
  - Tanh: Rango (-1,1)
- Loss functions:
  - Cross-Entropy: Clasificación
  - MSE: Regresión

**NO NECESITAS:** Derivaciones de gradientes, fórmulas complejas

---

### 🟡 UNIT 7: CNNs (3 preguntas)
**ESTUDIA (CONCEPTOS):**
- Convolución: Desliza filtro sobre imagen
- Pooling: Reduce dimensionalidad
- Feature maps: Qué aprenden
- Arquitecturas: LeNet, AlexNet, ResNet (nombres)
- Transfer Learning: Reutilizar modelos preentrenados

**NO NECESITAS:** Implementar CNN, matemática de convoluciones

---

### 🟡 UNIT 8: Transformers & RNNs (4 preguntas)
**ESTUDIA (SOLO CONCEPTOS):**
- RNNs: Memoria, LSTM, GRU (no implementar)
- Attention: Se enfoca en partes relevantes (concepto)
- Transformers:
  - Self-attention: Cada palabra ve todas las otras
  - Multi-head attention: Múltiples perspectivas
  - Positional encoding: Inyecta orden
- BERT, GPT: Nombres y usos

**NO NECESITAS:** Fórmulas de attention, código PyTorch

---

### 🟡 UNIT 9: Generative AI & LLMs (5 preguntas)
**ESTUDIA:**
- LLMs: Modelos que generan texto
- GPT: Autoregresivo, predice siguiente token
- BERT: Bidireccional, para classification
- Hugging Face: Librería para usar modelos preentrenados
- RAG (Retrieval Augmented Generation):
  - Retrieve: Busca en base de datos
  - Augment: Agrega al prompt
  - Generate: Genera respuesta
- Prompt Engineering: Cómo pedir al LLM
- GANs: Generador vs Discriminador (concepto)
- Diffusion Models: Denoise iterativo

**NO NECESITAS:** Entrenar LLMs, arquitecturas internas

---

### ⚪ UNIT 10: MLOps (1 pregunta, si entra)
**ESTUDIA:**
- Deployment: Llevar modelo a producción
- Monitoring: Vigilar rendimiento
- CI/CD: Automatizar pruebas
- Docker: Containerizar código

**PRIORIDAD BAJA** - Probablemente no entra en test teórico

---

## 📋 TABLA DE MEMORIZACIÓN (Units 6-10)

### Conceptos Clave Deep Learning

| Concepto | Qué es | Cuándo se usa |
|----------|--------|---------------|
| Perceptrón | Unidad básica, suma ponderada + activación | Bloque de todas las redes |
| MLP | Red con capas ocultas | Datos tabulares, regresión/clasificación |
| ReLU | max(0, x) | Activación en capas ocultas (estándar) |
| Sigmoid | 1/(1+e^(-x)) | Probabilidades (0,1) en output |
| Cross-Entropy | -Σ y*log(ŷ) | Loss para clasificación |
| Backprop | Propaga gradientes hacia atrás | Entrenar la red |
| CNN | Convoluciones + Pooling | Imágenes |
| RNN | Recurrencia, estado oculto | Secuencias (texto, series) |
| LSTM | Gates para memoria | RNN mejorado, secuencias largas |
| Attention | Pesa importancia de cada posición | Transformers, "enfoque" en relevante |
| Transformer | Attention + capas densas | NLP moderno (BERT, GPT) |
| LLM | Red neuronal masiva, predice tokens | Generar texto, Q&A |
| RAG | Retrieve + Augment + Generate | LLM con datos externos |

---

## 🚨 TRUCOS PARA PASAR EL TEST CON TODO

### 1. **Prioridad: Units 1-5**
- Si no sabes Unit 6-10 pero dominas 1-5 → Pasas (80% bien)
- Si no sabes Units 1-5 → Repruebas

### 2. **Para Units 6-10, memoriza NOMBRES y CONCEPTOS, no matemática**
- "¿Qué es un Transformer?" → "Red que usa attention"
- "¿Qué es RAG?" → "Retrieve documentos + Augment prompt + Generate respuesta"
- No necesitas: Fórmulas, implementación, detalles

### 3. **Estrategia en el test:**
- Preguntas Units 1-5: Responde con confianza
- Preguntas Units 6-10: Elimina opciones obviamente falsas
- Si no sabes: Busca en opciones algo que suene a lo que estudiaste

### 4. **Palabras clave por Unit:**

**Unit 6:** Perceptrón, MLP, Backpropagation, ReLU, Sigmoid, Cross-Entropy  
**Unit 7:** CNN, Convolución, Pooling, Transfer Learning  
**Unit 8:** RNN, LSTM, Attention, Transformer, Self-Attention  
**Unit 9:** LLM, GPT, BERT, RAG, Hugging Face, Prompt  
**Unit 10:** Deployment, Docker, Monitoring, CI/CD  

---

## 📅 CRONOGRAMA AGRESIVO (10 DÍAS)

### Días 1-2: UNITS 1-3 (Repaso rápido)
- [ ] Qué es AI, ML, DL (30 min)
- [ ] Paradigmas: Sup, NoSup, Refuerzo (30 min)
- [ ] Tres paradigmas básicos: KNN, KMeans, QL (1 hora)

### Días 2-3: UNIT 4 (Supervisado)
- [ ] Regresión (30 min)
- [ ] Clasificación (30 min)
- [ ] Árboles, RF, SVM (1 hora)
- [ ] Métricas (1 hora)

### Día 4: UNIT 5 (No supervisado)
- [ ] Clustering, PCA, Anomaly Detection (1.5 horas)
- [ ] Evaluación clustering (30 min)

### Días 5-6: UNIT 6 (Deep Learning Basics)
- [ ] Perceptrón (30 min)
- [ ] MLP y Backprop (1 hora)
- [ ] Activaciones y Loss (30 min)
- [ ] PyTorch basics (30 min)

### Día 7: UNIT 7 (CNNs)
- [ ] Convoluciones (30 min)
- [ ] Pooling (15 min)
- [ ] Transfer Learning (15 min)
- [ ] Arquitecturas nombrados (15 min)

### Día 8: UNIT 8 (Transformers)
- [ ] RNNs brevemente (15 min)
- [ ] Attention (30 min)
- [ ] Transformers (30 min)
- [ ] BERT, GPT (15 min)

### Día 9: UNIT 9 (LLMs)
- [ ] LLM basics (30 min)
- [ ] RAG (30 min)
- [ ] Hugging Face (30 min)
- [ ] GANs, Diffusion (30 min)

### Día 10: REPASO + DESCANSO
- [ ] Revisar temas débiles (1 hora)
- [ ] Dormir bien (8 horas)

---

## 🎯 OBJETIVOS REALISTAS

**No pretendas memorizar todo.** En su lugar:

✅ **SABER:**
- Nombres de algoritmos
- Para qué sirven
- Cuándo se usan
- Conceptos principales (sin matemática)

❌ **NO NECESITAS:**
- Derivaciones matemáticas
- Código detallado
- Implementación
- Detalles técnicos

---

## 💡 RATIO ESFUERZO-RECOMPENSA

| Unidad | Horas | Preguntas | Prioridad |
|--------|-------|-----------|-----------|
| Units 1-5 | 4 | ~80 | 🔴 MÁXIMA |
| Unit 6 | 2 | ~8 | 🟠 ALTA |
| Unit 7 | 1 | ~3 | 🟡 MEDIA |
| Unit 8 | 1.5 | ~4 | 🟡 MEDIA |
| Unit 9 | 1.5 | ~5 | 🟡 MEDIA |
| Unit 10 | 0.5 | ~1 | ⚪ BAJA |

**Total: ~11 horas en 10 días = MANEJABLE**

---

## ✅ CHECKLIST FINAL

**Antes del examen, debes poder:**

- [ ] Explicar regresión vs clasificación
- [ ] Matriz de confusión con TP, TN, FP, FN
- [ ] Cuándo usar Precision, Recall, F1
- [ ] Diferencia: Supervisado, No supervisado, Refuerzo
- [ ] Qué hace un árbol, RF, KNN, SVM
- [ ] Qué es overfitting y cómo detectarlo
- [ ] Qué es un Perceptrón y su limitación
- [ ] MLP y Backpropagation (concepto)
- [ ] CNN con convoluciones
- [ ] RNN y LSTM (básicamente)
- [ ] Transformers y Attention
- [ ] LLM y RAG
- [ ] Nombres de BERT, GPT, ResNet

---

**Si dominas esto ↑, pasas el examen.**

