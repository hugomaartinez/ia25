# Plan de Preparación - Examen Teórico (22 Mayo)

## 📊 Análisis de las 100 Preguntas

Basado en las 94 preguntas mostradas, el examen cubre principalmente **Units 1-5** (ML Clásico).

### Distribución por Tema

| Tema | % Estimado |
|------|---|
| **Métricas y Evaluación** | 12% |
| **Regresión** | 8% |
| **Clasificación** | 10% |
| **Árboles y Forests** | 7% |
| **Clustering & Unsupervised** | 6% |
| **Datos Faltantes y Limpieza** | 8% |
| **Train/Test/Validation** | 5% |
| **Q-Learning & RL** | 8% |
| **Normalización/Scaling** | 4% |
| **Pipelines y Validación** | 6% |
| **NumPy/Pandas** | 3% |
| **KNN** | 3% |
| **Paradigmas ML** | 5% |
| **Otros** | 5% |

---

## 🎯 CONCEPTOS CLAVE A DOMINAR (Por Prioridad)

### 🔴 CRÍTICO - MEMORIZA PRIMERO (30%)

#### 1. **Matriz de Confusión (Classification)**
```
TP = Verdadero Positivo   (predicho +, real +)
TN = Verdadero Negativo   (predicho -, real -)
FP = Falso Positivo       (predicho +, real -)  → Error Tipo I
FN = Falso Negativo       (predicho -, real +)  → Error Tipo II
```

**Métricas derivadas:**
- **Accuracy** = (TP+TN)/Total → ⚠️ Engañosa con clases desbalanceadas
- **Precision** = TP/(TP+FP) → "De mis predicciones +, cuántas son correctas"
- **Recall** = TP/(TP+FN) → "De los + reales, cuántos detecté"
- **F1** = 2×(Precision×Recall)/(Precision+Recall) → Balance

**Cuándo priorizar cada una:**
- **Precision Alto:** Filtro de spam (evitar falsos positivos)
- **Recall Alto:** Diagnóstico médico (detectar todos los casos)
- **F1:** Cuando necesitas balance

#### 2. **Métricas de Regresión**
- **MSE** = (Σ(y-ŷ)²)/n → Penaliza errores grandes
- **RMSE** = √MSE → Misma unidad que y
- **MAE** = Σ|y-ŷ|/n → Robusto a outliers
- **R²** = 1 - (SS_res/SS_tot) → Proporción de varianza explicada (0-1)

**Regla:**
- Use RMSE cuando errores grandes son inaceptables
- Use MAE cuando quiere robustez a outliers
- Use R² siempre para interpretar

#### 3. **Silhouette Score (Clustering)**
- Rango: -1 a +1
- **+1** = Clusters densos y bien separados
- **0** = Clusters solapados
- **-1** = Asignación errónea

---

### 🟠 MUY IMPORTANTE (25%)

#### 4. **Manejo de Datos Faltantes**

**Tres mecanismos:**
- **MCAR** (Missing Completely At Random): Causa aleatoria pura
  - Solución: Borrar es OK
  
- **MAR** (Missing At Random): Causa en variables observadas
  - Solución: Imputar es mejor
  
- **MNAR** (Missing Not At Random): Causa en el propio valor o no observado
  - Solución: Difícil, riesgo de sesgo

**Métodos de imputación:**
- Media: Rápido pero sesga varianza
- Mediana: Robusto a outliers
- KNN: Preserva correlaciones
- Árbol: Más preciso

**Riesgo de listwise deletion:** Pierdes información de otras columnas + sesgo si no MCAR

#### 5. **Normalización vs Estandarización**

```
StandardScaler:  x' = (x - mean) / std   → Rango indefinido
MinMaxScaler:    x' = (x - min) / (max - min) → [0,1]
```

**Diferencia clave:**
- MinMaxScaler es MUY sensible a outliers (el rango se estira)
- StandardScaler es más robusto

**Correcta aplicación:**
```python
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # NO fit aquí
```

#### 6. **Overfitting vs Underfitting**

| Síntoma | Causa | Solución |
|---------|-------|----------|
| error_train << error_test | Memorización (alta varianza) | Menos parámetros, regularización |
| error_train ≈ error_test (ambos altos) | Modelo simple (alto sesgo) | Más parámetros, modelo complejo |

**Detección:**
- Curva de aprendizaje: train baja, val sube → overfitting
- Train error mucho menor que test → overfitting

#### 7. **Data Leakage**

**Causas comunes:**
- ❌ Calcular estadísticas (media, min, max) EN TODO el dataset antes de dividir
- ❌ Usar Test set para seleccionar modelo o ajustar hiperparámetros
- ❌ Hacer `scaler.fit(X_test)`
- ✅ Reglas de negocio universales (edad > 150 es error) antes del split

---

### 🟡 IMPORTANTE (20%)

#### 8. **Paradigmas de ML**

**Supervisado (con etiquetas):**
- Regresión: Predicción continua → Métrica: MSE, RMSE, R²
- Clasificación: Predicción categórica → Métrica: Accuracy, Precision, Recall, F1
- Multietiqueta: Múltiples etiquetas simultáneas

**No Supervisado (sin etiquetas):**
- Clustering: Agrupar datos → K-Means, DBSCAN
- Dimensionalidad: Reducir features → PCA
- Anomaly Detection: Encontrar outliers

**Refuerzo:**
- Agente aprende interactuando con entorno
- Basado en recompensas

#### 9. **Train/Test/Validation Split**

```
Entrenamiento (60-70%): Entrenar el modelo
Validación (15%):       Ajustar hiperparámetros durante entrenamiento
Test (15-20%):          Evaluación final ÚNICA
```

**Orden correcto:**
1. Dividir datos
2. Fit scaler EN TRAIN
3. Transform TRAIN + TEST
4. Entrenar modelo EN TRAIN
5. Evaluar EN TEST (nunca antes)

#### 10. **Pipelines en Scikit-learn**

```python
Pipeline([
    ('scaler', StandardScaler()),        # Paso intermedio: TRANSFORMADOR
    ('model', LogisticRegression())      # Paso final: ESTIMADOR
])
```

**Requisito:**
- Pasos intermedios deben implementar `fit()` y `transform()`
- Paso final puede ser solo `fit()`

#### 11. **Validación Cruzada (K-Fold)**
- Divide train en K partes
- Entrena K veces (cada vez usa K-1 partes para train, 1 para val)
- Promedia resultados
- Costo: K veces más computación

---

### 🟢 MODERADO (15%)

#### 12. **Algoritmos Clave**

**Regresión Lineal:**
- Asume: y = β₀ + β₁X₁ + β₂X₂ + ...
- Minimiza: MSE
- Coeficiente β=0.5 → por cada unidad de X, y aumenta 0.5

**Logistic Regression:**
- Clasificación binaria
- Aplica sigmoide: σ(z) = 1/(1+e^(-z))
- Output: probabilidad [0,1]

**KNN (K-Nearest Neighbors):**
- Lazy learning (entrenamiento casi nulo)
- Predicción lenta (compara contra todos)
- Regresión: Promedia valores de K vecinos
- Clasificación: Votación mayoritaria

**Árboles de Decisión:**
- No paramétrico
- Robusto a outliers (usa umbrales, no medias)
- `max_depth`: Regularizador que previene overfitting
- Manejo nativo de faltantes (aprende dirección)

**Random Forest:**
- Múltiples árboles con bootstrap (muestreo con reemplazo)
- Bagging: Reduce varianza mediante promediado
- Clasificación: Votación mayoritaria
- Regresión: Promedio de predicciones

**K-Means:**
- Clustering no supervisado
- Minimiza: Inercia (WCSS = suma de distancias al centroide)
- Sensible a inicialización → usar `n_init` para múltiples ejecuciones
- Atributos: `labels_`, `cluster_centers_`, `inertia_`

**SVM:**
- Encuentra hiperplano que maximiza margen
- Support vectors: Puntos fronterizos que definen el margen

---

### 🟣 MODERADO (10%)

#### 13. **Q-Learning & Reinforcement Learning**

**Q-Table:**
- Filas = Estados, Columnas = Acciones
- Q(s,a) = Recompensa futura esperada
- Inicializada a ceros (el agente no sabe nada)

**Epsilon-Greedy (Balance Exploración-Explotación):**
- ε = Probabilidad de exploración aleatoria
- 1-ε = Probabilidad de usar mejor acción conocida
- ε decae durante entrenamiento (inicio: mucha exploración → final: explotación)

**Hiperparámetros:**
- **α (Learning Rate):** α=1 (rápido, inestable), α=0 (no aprende), típico: 0.1-0.5
- **γ (Discount Factor):** γ=0 (solo inmediato), γ=1 (importa futuro), típico: 0.9-0.99

**Fórmula:**
```
Q(s,a) ← Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
         └────────┘ └────────────────────────────┘
        Valor antiguo    TD Error (Temporal Difference)
```

---

## 📅 Cronograma de 10 Días

| Día | Tema | Horas |
|-----|------|-------|
| 1-2 | Métricas (matriz confusión, Accuracy, Precision, Recall, F1) | 2 |
| 2-3 | Regresión (MSE, RMSE, MAE, R²) + Métricas clustering | 2 |
| 3-4 | Datos faltantes (MCAR, MAR, MNAR) + Normalización | 2 |
| 4-5 | Overfitting, Data Leakage, Train/Test/Val split | 2 |
| 5-6 | Paradigmas ML + Pipelines + Validación Cruzada | 2 |
| 6-7 | Algoritmos: Regresión, KNN, Árboles, Random Forest | 3 |
| 7-8 | Algoritmos: K-Means, SVM, Encoding categorías | 2 |
| 8-9 | Q-Learning y Reinforcement Learning | 2 |
| 9 | Simulación de test (20 preguntas cronometradas) | 1 |
| 10 | Repaso últimas dudas + revisión general | 1 |

**Total: ~18 horas distribuidas en 10 días = 1.8 horas/día**

---

## ✅ CHECKLIST FINAL (Día antes del examen)

- [ ] Matriz de confusión memorizada
- [ ] Fórmulas de métricas claras
- [ ] MCAR vs MAR vs MNAR diferenciado
- [ ] StandardScaler vs MinMaxScaler
- [ ] Train/Test/Val split proceso correcto
- [ ] 6-7 algoritmos principales claros
- [ ] Q-Learning conceptos básicos
- [ ] Hice simulacro de 20 preguntas
- [ ] Identifiqué 3 temas débiles y repasé

---

**Última actualización:** 12 Mayo 2026  
**Examen Teórico:** 22 Mayo 2026 (10 días)

