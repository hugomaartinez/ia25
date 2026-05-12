# 📚 Guía de Preparación para Examen - IA 2026

## 🎯 Resumen Rápido

**Examen:** Martes 19 de Mayo 2026
**Formato:** Sin internet, pero puedes llevar archivos
**Contenido:** Machine Learning Clásico + Deep Learning

---

## 📁 Archivos Para Llevar al Examen

**Imprime estos 3 archivos y llévalos (PDFs o impresos):**

1. **EXAM_CONTENT_STRUCTURE.md**
   - Qué entra en cada parcial
   - Referencia rápida de temas

2. **EXAM_QUICK_REFERENCE.md** ⭐ PRINCIPAL
   - Workflow paso a paso
   - Pandas + Numpy + Sklearn
   - Métricas y cuándo usarlas
   - Code snippets listos para copiar

3. **EXAM_DEEP_LEARNING_REFERENCE.md** ⭐ PARA 2DO PARCIAL
   - PyTorch basics
   - CNNs
   - Transformers
   - Hugging Face

---

## 🎓 Estructura del Examen (Estimado)

### Primer Parcial (YA HECHO) ✅
- Units 1-5: ML Clásico
- Proyectos: Regresión, Clasificación, Clustering

### Segundo Parcial (19 MAYO) 🎯
- Units 6-10: Deep Learning + NLP + MLOps
- Preguntas esperadas:
  - Entrenar red neuronal simple (PyTorch)
  - Clasificar imágenes (CNN)
  - Procesar texto (Transformers/Hugging Face)
  - Posible pregunta sobre RAG

---

## 🏃 Plan de Estudio (Óptimo)

```
DÍA 1-2: PyTorch basics + MLPs
DÍA 3-4: CNNs (MNIST, CIFAR-10)
DÍA 5:   Transformers (BERT, GPT conceptos)
DÍA 6:   RAG + Hugging Face práctico
DÍA 7:   Simulación de examen (dataset nuevo)
```

---

## 🔗 Recursos en el Repo

### Deep Learning Notebooks
- `pytorch/01-deep-learning-fundamentals/` - **ESENCIAL**
- `pytorch/02-computer-vision-cnn/` - **ESENCIAL**

### Machine Learning (Repaso)
- `sklearn/` - Todos los proyectos base
- `algorithms/` - Teoría de cada algoritmo
- `background/` - Conceptos fundamentales

### LLMs (Para RAG)
- README cita: `https://github.com/A-nit-A/IES_Teis_AI_Triage`
- Busca ejemplos de ChromaDB + embeddings

---

## ✅ Checklist Pre-Examen

**1 Semana Antes:**
- [ ] He impreso/guardado los 3 archivos de referencia
- [ ] He ejecutado todos los notebooks de PyTorch
- [ ] He entrenado al menos 2 redes neuronales desde cero
- [ ] He probado Hugging Face (clasificación, generación)

**3 Días Antes:**
- [ ] He entrenado CNN en MNIST
- [ ] He entendido transfer learning
- [ ] He visto ejemplo de RAG funcionando

**1 Día Antes:**
- [ ] He resuelto 3 problemas sin referencia (solo mis archivos)
- [ ] He anotado errores comunes
- [ ] He revisado temas débiles

**Mañana del Examen:**
- [ ] Tengo los 3 archivos listos
- [ ] Llevo laptop con PyTorch instalado
- [ ] He revisado mi "cheat sheet" personal (si creaste una)

---

## 🧠 Conceptos Clave a Dominar

### Imprescindible:
1. ✅ PyTorch: tensor operations + nn.Module + training loop
2. ✅ CNNs: Conv2d, pooling, how features are learned
3. ✅ Transformers: Self-attention concept (no necesitas implementar)
4. ✅ Hugging Face: Cómo cargar y usar modelos preentrenados

### Importante:
1. ⚠️ Cross-entropy loss vs MSE
2. ⚠️ Backpropagation conceptualmente
3. ⚠️ Overfitting detection
4. ⚠️ RAG pipeline básico

### Menos Crítico:
- Detalles matemáticos de attention
- ONNX, TensorFlow (aunque puede entrar)
- MLOps production details

---

## 💡 Tips Importantes

### Durante el Examen:
1. **Identifica tipo de problema:** Regresión, clasificación, generación, etc.
2. **Prepara datos primero:** EDA → Limpieza → Normalización
3. **Elije modelo apropiado:** 
   - Tabular + labels → sklearn
   - Imágenes → CNN con PyTorch
   - Texto → Hugging Face Transformers
4. **Evalúa correctamente:** Usa métricas relevantes
5. **Documenta supuestos:** Explica decisiones

### Si Algo No Funciona:
- No entres en pánico
- Usa tus archivos de referencia
- Simplifica: empieza con algo mínimo que funcione
- Incrementa complejidad gradualmente

---

## 🚀 Después del Examen

Una vez termines el examen (sea bien o no):
1. Anota qué preguntas entrada
2. Guarda soluciones en el repo
3. Documenta patrones nuevos
4. Actualiza esta guía para futuros estudiantes

---

## 📝 Notas sobre Restauración

El archivo original `todos/wip-roadmap.md` fue eliminado pero está disponible en git history. 
Este documento sustituye esa información de forma **práctica** para el examen.

**Última compilación:** 12 de Mayo 2026
**Examen:** 19 de Mayo 2026

---

## ¿Preguntas Frecuentes?

**P: ¿Necesito memorizar fórmulas?**
A: No. Necesitas saber QUÉ hace cada algoritmo, no la fórmula exacta.

**P: ¿Entra MLOps?**
A: Posiblemente preguntas básicas (Docker, CI/CD conceptos). Pero enfócate en algorítmica.

**P: ¿Qué si no hay internet?**
A: Por eso llevas los archivos PDF. Puedes programar sin internet.

**P: ¿Timeout en modelos grandes?**
A: Usa modelos pequeños ('distilbert' en lugar de 'bert-large'). Pero probablemente dan un dataset pequeño.

**P: ¿Cuánto tiempo dedico a cada tema?**
A: PyTorch 30%, CNNs 25%, Transformers 25%, RAG 10%, Repaso 10%

---

Buena suerte en el examen 🚀
