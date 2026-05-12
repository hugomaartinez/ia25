# 🎓 EXAMEN COMPLETO - Units 1-10 (TODO EL CURSO)

**Ambos exámenes cubren Units 1-10**
- 22 Mayo (Teórico): Test conceptos Units 1-10
- 26 Mayo (Práctico): Código Units 1-10

---

# PARTE 1: EJERCICIOS PRÁCTICOS DEEP LEARNING

---

## TIPO 1: ENTRENAR RED NEURONAL SIMPLE EN PYTORCH

### Enunciado Típico:
"Entrena MLP en MNIST. Reporta accuracy."

### Solución:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score

# PASO 1: PREPARAR DATOS
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# PASO 2: DEFINIR ARQUITECTURA
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)      # Input: 784 (28×28)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)          # Output: 10 clases
    
    def forward(self, x):
        x = x.view(-1, 28*28)                 # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

model = SimpleNN()

# PASO 3: CONFIGURAR ENTRENAMIENTO
loss_fn = nn.CrossEntropyLoss()              # Para clasificación
optimizer = optim.Adam(model.parameters(), lr=0.001)

# PASO 4: ENTRENAR
epochs = 5
for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        # Forward pass
        output = model(batch_X)
        loss = loss_fn(output, batch_y)
        
        # Backward pass
        optimizer.zero_grad()                 # Limpiar gradientes
        loss.backward()                       # Calcular gradientes
        optimizer.step()                      # Actualizar pesos
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# PASO 5: EVALUAR
model.eval()  # Modo evaluación (desactiva dropout, etc)
all_preds = []
all_labels = []

with torch.no_grad():  # No calcular gradientes
    for batch_X, batch_y in test_loader:
        output = model(batch_X)
        preds = output.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"\nTest Accuracy: {accuracy:.4f}")

# PASO 6: DETECTAR OVERFITTING
# Entrenar también en train set y comparar
model.train()
train_preds = []
with torch.no_grad():
    for batch_X, batch_y in train_loader:
        output = model(batch_X)
        preds = output.argmax(dim=1)
        train_preds.extend(preds.cpu().numpy())

train_acc = accuracy_score([y for _, y in train_dataset], train_preds)
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {accuracy:.4f}")

if train_acc - accuracy > 0.1:
    print("⚠️ POSIBLE OVERFITTING")
```

### Checklist:
- ✅ Carga datos con DataLoader
- ✅ Normaliza imágenes
- ✅ Define arquitectura (capas, activaciones)
- ✅ Usa loss correcto (CrossEntropyLoss para clasificación)
- ✅ Loop de entrenamiento: forward → backward → step
- ✅ Evalúa en test set
- ✅ Detecta overfitting comparando train vs test

---

## TIPO 2: CNN PARA IMÁGENES

### Enunciado Típico:
"Crea CNN para CIFAR-10. Reporta accuracy."

### Solución:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# PASO 1: DATOS (con augmentación)
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# PASO 2: ARQUITECTURA CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Convolucional 1: 3 canales → 32 filtros
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 32×32 → 16×16
        
        # Convolucional 2: 32 filtros → 64 filtros
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 16×16 → 8×8
        
        # Fully connected
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)      # (batch, 3, 32, 32) → (batch, 32, 32, 32)
        x = self.relu1(x)
        x = self.pool1(x)      # (batch, 32, 32, 32) → (batch, 32, 16, 16)
        
        x = self.conv2(x)      # (batch, 32, 16, 16) → (batch, 64, 16, 16)
        x = self.relu2(x)
        x = self.pool2(x)      # (batch, 64, 16, 16) → (batch, 64, 8, 8)
        
        x = x.view(x.size(0), -1)  # Flatten: (batch, 64*8*8)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = SimpleCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# PASO 3: ENTRENAR (igual que MLP)
epochs = 5
for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        output = model(batch_X)
        loss = loss_fn(output, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# PASO 4: EVALUAR
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        output = model(batch_X)
        preds = output.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")
```

### Conceptos clave:
- **Conv2d:** Desliza filtro sobre imagen
- **MaxPool2d:** Reduce dimensionalidad (2×2 → 1)
- **Flatten:** Convierte (batch, canales, H, W) → (batch, features)
- **Dropout:** Regularización para prevenir overfitting

---

## TIPO 3: TRANSFER LEARNING

### Enunciado Típico:
"Usa ResNet preentrenada para clasificar imágenes nuevas."

### Solución:

```python
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

# PASO 1: CARGAR MODELO PREENTRENADO
model = models.resnet18(pretrained=True)

# PASO 2: CONGELAR PESOS (NO entrenar características base)
for param in model.parameters():
    param.requires_grad = False

# PASO 3: REEMPLAZAR CAPA FINAL
num_classes = 10  # Tu dataset tiene 10 clases
model.fc = nn.Linear(model.fc.in_features, num_classes)

# PASO 4: DESCONGELAR ÚLTIMAS CAPAS (opcional, para fine-tuning)
for param in model.layer4.parameters():
    param.requires_grad = True

# PASO 5: ENTRENAR (solo capa final)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# PASO 6: ENTRENAR (código similar a MLP/CNN anterior)
for epoch in range(5):
    for batch_X, batch_y in train_loader:
        output = model(batch_X)
        loss = loss_fn(output, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# PASO 7: EVALUAR
model.eval()
# ... (código de evaluación)

# VENTAJA: ResNet ya aprendió features útiles de ImageNet
# Solo necesitas entrenar la última capa (rápido)
```

### Cuándo usar:
- Dataset pequeño (<10k imágenes)
- No tienes GPU potente
- Datos similares a ImageNet (fotografías naturales)

---

## TIPO 4: USAR HUGGING FACE (TRANSFORMERS)

### Enunciado Típico:
"Clasifica sentimiento de textos usando BERT preentrenado."

### Solución:

```python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TextClassificationPipeline
import torch

# OPCIÓN 1: FÁCIL - Usar Pipeline
classifier = TextClassificationPipeline(
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Clasificar textos
texts = [
    "This movie is amazing!",
    "I hate this film"
]

results = classifier(texts)
for text, result in zip(texts, results):
    print(f"{text}")
    print(f"  Label: {result['label']} (score: {result['score']:.4f})\n")

# Output:
# This movie is amazing!
#   Label: POSITIVE (score: 0.9998)
# I hate this film
#   Label: NEGATIVE (score: 0.9999)

# OPCIÓN 2: MANUAL - Control fino
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2  # Binaria: positivo/negativo
)

# Tokenizar
text = "This movie is great"
tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

# Predecir
with torch.no_grad():
    output = model(**tokens)
    logits = output.logits
    prediction = logits.argmax(dim=1)
    confidence = torch.softmax(logits, dim=1)

print(f"Prediction: {prediction.item()}")
print(f"Confidence: {confidence[0].tolist()}")

# OPCIÓN 3: FINE-TUNING (si tienes datos propios)
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments

# 1. Crear dataset personalizado
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx])
        }

# 2. Crear dataset
train_texts = ["great movie!", "terrible film", ...]
train_labels = [1, 0, ...]
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)

# 3. Configurar entrenamiento
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
)

# 4. Entrenar
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

# 5. Usar modelo entrenado
model.eval()
# ... predecir con modelo entrenado
```

### Cuándo usar cada opción:
- **Pipeline:** Rápido, solo predicción, modelo preentrenado
- **Manual:** Control fino, entender qué pasa
- **Fine-tuning:** Datos propios, mejor rendimiento

---

# PARTE 2: PREGUNTAS TEÓRICAS UNITS 6-10

---

## UNIT 6: DEEP LEARNING FUNDAMENTALS

### P1: ¿Qué es un Perceptrón?

**Definición:** Unidad básica de red neuronal (inspirada en neuronas biológicas)

**Componentes:**
1. Entradas ponderadas: x₁w₁ + x₂w₂ + ... + xₙwₙ
2. Suma ponderada + sesgo: z = Σ(xᵢwᵢ) + b
3. Función de activación: a = f(z)

**Limitación fundamental:** 
- Solo puede resolver problemas **linealmente separables**
- Ejemplo: No puede aprender XOR

**Símbolo:**
```
x₁ ──┐
     ├──→ [Σ] ──→ [f(z)] ──→ output
x₂ ──┤
```

---

### P2: ¿Qué es MLP (Multi-Layer Perceptron)?

**Definición:** Red con múltiples capas de perceptrones

**Capas:**
1. **Input Layer:** Datos
2. **Hidden Layers:** 1 o más capas ocultas (introducen no-linealidad)
3. **Output Layer:** Predicción

**Diferencia con Perceptrón simple:**
- Perceptrón: 1 capa → problemas lineales
- MLP: Capas ocultas → problemas no-lineales

**Ejemplo arquitectura:**
```
Input (784)
    ↓
Hidden (128)  ← Capa oculta 1
    ↓
Hidden (64)   ← Capa oculta 2
    ↓
Output (10)   ← Clasificación
```

---

### P3: ¿Qué es Backpropagation?

**Proceso de 4 pasos:**

1. **Forward Pass:** Datos pasan adelante, calcular predicción ŷ
2. **Calcular Error:** error = y - ŷ (predicción vs real)
3. **Backward Pass:** Propagar error hacia atrás usando regla de la cadena
4. **Actualizar Pesos:** Mover pesos en dirección que reduce error

**Algoritmo (simplificado):**
```
1. w_nuevo = w_viejo - α × ∂Loss/∂w
            └──────────────────────┘
                Gradiente
```

**Intuición:** Como bajar una montaña siguiendo la pendiente más pronunciada

---

### P4: ¿Cuál es la diferencia entre ReLU, Sigmoid y Tanh?

| Activación | Fórmula | Rango | Uso | Ventaja |
|-----------|---------|-------|-----|---------|
| ReLU | max(0, x) | [0, ∞) | Capas ocultas (ESTÁNDAR) | Rápido, no vanishing gradient |
| Sigmoid | 1/(1+e^(-x)) | [0, 1] | Output clasificación binaria | Probabilidad |
| Tanh | (e^x - e^(-x))/(e^x + e^(-x)) | [-1, 1] | Capas ocultas | Centrado en 0 |

**Cuándo usar cada una:**
- **ReLU en capas ocultas:** Estándar moderno
- **Sigmoid en output:** Clasificación binaria
- **Softmax en output:** Multiclase (no listado pero importante)

---

### P5: ¿Cross-Entropy Loss vs MSE Loss?

**Cross-Entropy Loss:**
- Para clasificación
- Fórmula: -Σ y*log(ŷ) + (1-y)*log(1-ŷ)
- Penaliza predicciones confidentes pero incorrectas

**MSE Loss:**
- Para regresión
- Fórmula: (1/n) × Σ(y - ŷ)²
- Penaliza diferencia al cuadrado

**Cuándo usar:**
- Clasificación → Cross-Entropy
- Regresión → MSE

---

### P6: ¿Qué es Dropout?

**Definición:** Durante entrenamiento, desactiva neuronas aleatoriamente (probabilidad ε)

**Efecto:**
- Previene co-adaptación de neuronas
- Reduce overfitting

**En código:**
```python
self.dropout = nn.Dropout(0.5)  # 50% desactivadas
```

**Importante:** Dropout solo activo en TRAINING, no en evaluación

---

### P7: ¿Batch Normalization?

**Idea:** Normalizar inputs de cada capa durante entrenamiento

**Efecto:**
- Entrena más rápido
- Permite learning rates más altos
- Reduce overfitting

---

## UNIT 7: CONVOLUTIONAL NEURAL NETWORKS (CNNs)

### P8: ¿Qué es una Convolución?

**Idea:** Deslizar un filtro (kernel) sobre la imagen

**Proceso:**
1. Posiciona filtro en la imagen
2. Multiplica elemento a elemento
3. Suma resultado (esto es 1 valor del feature map)
4. Desliza a siguiente posición

**Resultado:** Feature map (imagen más pequeña)

**Ejemplo visual:**
```
Imagen 3×3       Filtro 3×3      Resultado
[1 2 3]          [0 1 0]         
[4 5 6]  ×       [1 0 1]   =     valor = 1×0 + 2×1 + 3×0 + 4×1 + 5×0 + ... = 5
[7 8 9]          [0 1 0]
```

---

### P9: ¿Pooling?

**Idea:** Reduce dimensionalidad manteniendo features importantes

**MaxPooling (más común):**
```
[1 5]  → 5    (coge el máximo)
[3 2]
```

**Efecto:**
- Imagen 32×32 → Imagen 16×16 (Max Pool 2×2)
- Reduce cálculos
- Aumenta receptive field

---

### P10: ¿Arquitectura CNN típica?

```
Input (imágenes)
    ↓
Conv2d + ReLU + MaxPool2d
    ↓
Conv2d + ReLU + MaxPool2d
    ↓
Flatten
    ↓
FC + ReLU
    ↓
FC (output)
```

**Cada capa aprende features progresivamente:**
- Layer 1: Bordes simples
- Layer 2: Formas
- Layer 3: Patrones complejos
- FC: Clasificación

---

### P11: ¿Transfer Learning?

**Idea:** Usar modelo preentrenado + ajustar para tu tarea

**Pasos:**
1. Cargar modelo preentrenado (ej: ResNet de ImageNet)
2. Congelar capas iniciales (features genéricas)
3. Reemplazar última capa (para tus clases)
4. Entrenar solo última capa

**Ventaja:**
- Menos datos necesarios
- Entrenamiento rápido
- Mejor rendimiento

---

## UNIT 8: TRANSFORMERS & SEQUENTIAL DATA

### P12: ¿RNN (Recurrent Neural Network)?

**Idea:** Procesa secuencias manteniendo estado/memoria

**Componente clave:** Conexión recurrente
```
      ↓
[RNN Cell] → output
      ↑
    (alimenta entrada siguiente)
```

**Problema:** Vanishing Gradient (gradientes se desvanecen con secuencias largas)

---

### P13: ¿LSTM (Long Short-Term Memory)?

**Mejora sobre RNN:** Usa "gates" para controlar flujo de información

**Gates:**
1. **Forget Gate:** Qué olvida de la memoria anterior
2. **Input Gate:** Qué información nueva agrega
3. **Output Gate:** Qué información expone

**Ventaja:** Puede aprender dependencias a largo plazo

---

### P14: ¿Attention Mechanism?

**Problema:** En seq2seq, la información se "olvida" en secuencias largas

**Solución:** Attention permite que el modelo "mire" atrás a toda la secuencia

**Idea:** 
- Para cada palabra en output, calcula "peso" de importancia para cada palabra en input
- Usa esos pesos para construir contexto

**Ejemplo:** Traducción inglés-español
```
Input:  "The cat sat on the mat"
         ↑ ↑            ↑
Output: "El gato se sentó sobre la alfombra"
```
Al generar "gato", atiende principalmente a "cat"

---

### P15: ¿Transformer?

**Revolución:** Elimina recurrencia, usa SOLO attention

**Componentes:**
1. **Self-Attention:** Cada palabra mira relación con todas las otras
2. **Multi-Head Attention:** Múltiples perspectivas simultáneamente
3. **Positional Encoding:** Inyecta información del orden
4. **Feed-Forward:** Red densa después de attention

**Ventaja:** Paralelizable (rápido), captura dependencias a largo plazo

**Arquitectura:**
```
Input
  ↓
Positional Encoding
  ↓
Multi-Head Self-Attention
  ↓
Feed-Forward Network
  ↓
Output
```

---

### P16: ¿BERT vs GPT?

**BERT:**
- **Bidireccional:** Ve contexto izquierda Y derecha
- Usa encoder de Transformer
- Mejor para: Clasificación, Q&A, análisis de texto
- Ejemplo: Entender significado de palabra en contexto

**GPT:**
- **Unidireccional:** Solo ve contexto a la izquierda
- Usa decoder de Transformer
- Mejor para: Generación de texto, chat
- Ejemplo: Completar texto, escribir ensayos

---

## UNIT 9: GENERATIVE AI & LLMS

### P17: ¿Qué es un LLM (Large Language Model)?

**Definición:** Red neuronal masiva entrenada en enormes cantidades de texto

**Funcionamiento:** Predice el siguiente token (palabra/subpalabra) basado en anteriores

**Ejemplo:**
```
Input:  "El gato se sentó en..."
Output: "la alfombra" (probabilidad más alta)
```

**Tamaño:**
- GPT-2: 1.5 mil millones parámetros
- GPT-3: 175 mil millones parámetros
- LLaMA 2: 7-70 mil millones parámetros

---

### P18: ¿Cómo genera texto un LLM?

**Proceso:**
1. **Tokenizar:** Convertir texto a números
2. **Forward pass:** Calcular probabilidades de siguiente token
3. **Sample:** Elegir token (o usar greedy: el de mayor prob)
4. **Repetir:** Usar token generado como entrada siguiente

**Ejemplo:**
```
Token 1: "El" → [0.1, 0.05, 0.7, 0.05, ...] → "gato" (prob 0.7)
Token 2: "El gato" → [0.02, 0.01, 0.05, 0.8, ...] → "se"
Token 3: "El gato se" → ...
```

---

### P19: ¿Hugging Face?

**Librería:** Simplifica uso de transformers preentrenados

**Pipeline (Fácil):**
```python
classifier = pipeline("sentiment-analysis")
result = classifier("I love this!")
# → {label: 'POSITIVE', score: 0.9999}
```

**Manual (Control):**
```python
tokenizer = AutoTokenizer.from_pretrained("bert-base")
model = AutoModelForSequenceClassification.from_pretrained("bert-base")
# ... más control sobre proceso
```

---

### P20: ¿RAG (Retrieval Augmented Generation)?

**Problema:** LLMs alucinan, dan información incorrecta, no saben de datos recientes

**Solución RAG:**
```
1. RETRIEVE: Busca documentos relevantes en base de datos
2. AUGMENT: Agrega documentos al prompt
3. GENERATE: LLM genera respuesta basada en documentos
```

**Ejemplo:** Chatbot de empresa
```
Pregunta: "¿Cuál es la política de vacaciones?"

1. Retrieve: Busca en manual de políticas → Encuentra doc
2. Augment: Agrega doc al prompt:
   "Basándote en este manual...[documento]..., contesta la pregunta"
3. Generate: LLM lee manual y responde correctamente
```

**Ventaja:** Información actualizada, específica de dominio, sin reentrenar modelo

---

### P21: ¿GANs (Generative Adversarial Networks)?

**Dos redes compitiendo:**
- **Generador:** Crea imágenes falsas
- **Discriminador:** Intenta detectar falsas vs reales

**Proceso:**
1. Generador genera imagen falsa
2. Discriminador intenta clasificarla
3. Feedback para ambas redes
4. Iteran hasta que discriminador no puede distinguir

**Aplicaciones:** Síntesis de imágenes, deepfakes, data augmentation

---

### P22: ¿Diffusion Models?

**Idea:** Agregar ruido gradualmente, aprender a revertirlo

**Proceso:**
1. **Forward:** Imagen limpia → Ruido gradual
2. **Reverse:** Ruido → Imagen limpia (lo que aprende)

**Ventajas sobre GANs:**
- Más estables para entrenar
- Mejor calidad de imágenes
- Mejor diversidad

**Ejemplos:** DALL-E 2, Stable Diffusion, Midjourney

---

## UNIT 10: MLOPS (Producción)

### P23: ¿Qué es MLOps?

**Definición:** DevOps aplicado a Machine Learning

**Ciclo:**
```
Data Collection → Preprocessing → Model Training → Deployment → Monitoring → Retrain
                                    ↑_________________↓
```

**Diferencia con Software tradicional:**
- Software tradicional: Código + bugs = fix
- ML: Código + datos + modelo + datos nuevo = puede degradar

---

### P24: ¿Deployment?

**Opciones:**

1. **Batch Prediction:**
   - Procesa millones de predicciones juntas
   - Resultado guardado en base de datos
   - Ejemplo: Recomendaciones nocturnas

2. **Real-time API:**
   - Expone modelo como API REST
   - Predice cuando se solicita
   - Latencia crítica
   - Framework: FastAPI

3. **Edge Deployment:**
   - Modelo en dispositivo (teléfono, IoT)
   - No requiere conexión
   - Ejemplo: TensorFlow Lite en móvil

---

### P25: ¿Monitoring?

**Métricas a vigilar:**
1. **Performance:** Accuracy, latency, throughput
2. **Data Drift:** ¿Los datos actuales son diferentes a entrenamiento?
3. **Model Drift:** ¿El modelo degradó?
4. **Predictions:** ¿Las predicciones son razonables?

**Consecuencia:** Si degrada → Reentrenar

---

---

# PARTE 3: ESTRATEGIA FINAL PARA 10 DÍAS

---

## TABLA DE PRIORIDADES

| Unit | Tema | Horas | Prioridad | Método |
|------|------|-------|-----------|--------|
| 1-2 | Fundamentos IA | 0.5 | ⚪ Baja | Repaso rápido |
| 3 | 3 Paradigmas | 1 | 🟡 Media | Recordar conceptos |
| 4-5 | ML Clásico | 4 | 🔴 CRÍTICO | DOMINAR |
| 6 | DL Basics | 2 | 🟠 Alta | Conceptos + código básico |
| 7 | CNNs | 1 | 🟡 Media | Conceptos + código MNIST |
| 8 | Transformers | 1 | 🟡 Media | Conceptos (no derivaciones) |
| 9 | LLMs | 1 | 🟡 Media | Conceptos + usar Hugging Face |
| 10 | MLOps | 0.5 | ⚪ Baja | Conceptos básicos |

**Total: ~11 horas**

---

## CRONOGRAMA AGRESIVO

### DÍAS 1-2: Units 1-5 (Repaso ML Clásico)
- [ ] Revisa EXAM_COMPLETE_SOLUTIONS.md (preguntas 1-42)
- [ ] Practica 2-3 problemas: regresión, clasificación, clustering
- [ ] Memoriza matriz confusión, métricas
- **Tiempo:** 4 horas

### DÍAS 3-4: Unit 6 (Deep Learning Basics)
- [ ] Lee preguntas P1-P7 (Unit 6)
- [ ] Ejecuta código de MLP en MNIST
- [ ] Entiende forward pass y backpropagation (concepto)
- **Tiempo:** 2 horas

### DÍA 5: Unit 7 (CNNs)
- [ ] Lee preguntas P8-P11
- [ ] Entiende convoluciones (concepto)
- [ ] Ejecuta código CNN en CIFAR-10
- **Tiempo:** 1 hora

### DÍA 6: Unit 8 (Transformers)
- [ ] Lee preguntas P12-P16
- [ ] Entiende Attention y Transformers (alto nivel)
- [ ] NO necesitas memorizar fórmulas
- **Tiempo:** 1 hora

### DÍA 7: Unit 9 (LLMs)
- [ ] Lee preguntas P17-P22
- [ ] Prueba Hugging Face (clasificación de sentimientos)
- [ ] Entiende RAG (concepto)
- **Tiempo:** 1 hora

### DÍA 8: Unit 10 (MLOps)
- [ ] Lee preguntas P23-P25
- [ ] Conceptos básicos (Deployment, Monitoring)
- **Tiempo:** 0.5 horas

### DÍAS 9-10: SIMULACRO + REPASO
- [ ] Haz test simulado (20 preguntas aleatorias)
- [ ] Identifica temas débiles
- [ ] Repasa esos temas
- [ ] Descansa bien día anterior
- **Tiempo:** 1.5 horas + descanso

---

## 🎯 QUÉ MEMORIZAR DE CADA UNIT

### Unit 6: Deep Learning
- ✅ Perceptrón: qué es, limitación (solo lineal)
- ✅ MLP: múltiples capas = no-lineal
- ✅ Backpropagation: proceso (forward → error → backward → update)
- ✅ ReLU para capas ocultas, Sigmoid para output binaria
- ✅ CrossEntropyLoss para clasificación

### Unit 7: CNNs
- ✅ Convolución: deslizar filtro sobre imagen
- ✅ Pooling: reduce dimensionalidad
- ✅ Feature maps: imagen transformada por filtro
- ✅ Transfer learning: usa modelo preentrenado

### Unit 8: Transformers
- ✅ Attention: modelo "mira" partes relevantes
- ✅ Self-attention: cada palabra ve todas las otras
- ✅ Multi-head: múltiples perspectivas
- ✅ Transformer: encoder (BERT) + decoder (GPT)

### Unit 9: LLMs
- ✅ LLM: predice siguiente token
- ✅ RAG: Retrieve → Augment → Generate
- ✅ Hugging Face: librería para usar modelos
- ✅ GANs vs Diffusion: dos tipos generativos

### Unit 10: MLOps
- ✅ Deployment: batch vs real-time
- ✅ Monitoring: vigilar performance y drift
- ✅ Data drift: cambios en datos nuevos

---

## ⚠️ NO MEMORICES

- ❌ Fórmulas matemáticas detalladas
- ❌ Derivaciones de backpropagation
- ❌ Fórmulas de attention
- ❌ Arquitecturas internas de transformers
- ❌ Detalles de implementación

---

## 💡 TIPS FINALES

1. **Examen Teórico (22 mayo):** 50% Units 1-5, 50% Units 6-10
   - Units 1-5 son "fácil dinero" (ya los viste)
   - Units 6-10 necesitas conceptos, no detalles

2. **Examen Práctico (26 mayo):** Probablemente igual contenido pero código
   - Necesitas saber escribir PyTorch, scikit-learn, Hugging Face
   - No es suficiente entender conceptos

3. **Distribución de esfuerzo:** 
   - 40% Units 1-5 (repaso)
   - 60% Units 6-10 (nuevo)

4. **Lo que probablemente NO entra en test:**
   - Implementar Transformer desde cero
   - Entrenar LLM desde cero
   - MLOps avanzado

5. **Lo que SÍ entra en test:**
   - Usar Hugging Face preentrenado
   - Conceptos de CNN, Attention, RAG
   - Código PyTorch básico
   - Usar Transfer Learning

---

## 📊 RESUMEN RÁPIDO - CHEAT SHEET

### Métrica Correcta
- **Regresión:** MSE, RMSE, R²
- **Clasificación:** Accuracy (si balanceada), Precision/Recall/F1 (si desbalanceada)
- **Clustering:** Silhouette Score

### Modelo Correcto
- **Pocos datos, tabular:** Árbol o RandomForest
- **Muchos datos, tabular:** Logistic Regression o Neural Network
- **Imágenes:** CNN
- **Texto:** Transformer (BERT/GPT)
- **Sin etiquetas:** Clustering (K-Means) o Anomaly Detection

### Prevenir Overfitting
- Más datos
- Modelo más simple (max_depth ↓, regularización ↑)
- Dropout
- Early stopping

### Data Leakage
- Normalizar DESPUÉS de train/test split
- Seleccionar features SOLO de train set
- No usar test set para ajustar hiperparámetros

