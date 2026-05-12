# 🧠 Deep Learning Quick Reference - Segundo Parcial

---

## 1️⃣ PYTORCH BÁSICO

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Crear tensor
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.zeros((3, 4))
z = torch.randn((2, 3))  # Distribución normal

# Operaciones
x + y
x * 2
torch.matmul(x, y)
```

---

## 2️⃣ CONSTRUIR RED NEURONAL SIMPLE

```python
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Crear modelo
model = SimpleNN(input_size=28*28, hidden_size=128, output_size=10)

# Loss y optimizer
loss_fn = nn.CrossEntropyLoss()  # Para clasificación
# loss_fn = nn.MSELoss()          # Para regresión
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

---

## 3️⃣ LOOP DE ENTRENAMIENTO

```python
epochs = 10

for epoch in range(epochs):
    for batch in data_loader:
        X_batch, y_batch = batch
        
        # Forward pass
        output = model(X_batch)
        loss = loss_fn(output, y_batch)
        
        # Backward pass
        optimizer.zero_grad()      # Limpiar gradientes
        loss.backward()            # Calcular gradientes
        optimizer.step()           # Actualizar pesos
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")
```

---

## 4️⃣ REDES CONVOLUCIONALES (CNNs)

### Arquitectura básica:
```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Capas convolucionales
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)    # 1 canal → 32 filtros
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)                             # Reduce tamaño a la mitad
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)   # 32 → 64 filtros
        
        # Capas fully connected
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # x shape: (batch_size, 1, 28, 28) para MNIST
        
        x = self.conv1(x)          # → (batch_size, 32, 28, 28)
        x = self.relu(x)
        x = self.pool(x)           # → (batch_size, 32, 14, 14)
        
        x = self.conv2(x)          # → (batch_size, 64, 14, 14)
        x = self.relu(x)
        x = self.pool(x)           # → (batch_size, 64, 7, 7)
        
        x = x.view(x.size(0), -1)  # Flatten: → (batch_size, 64*7*7)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)            # → (batch_size, 10)
        
        return x
```

### Conceptos clave CNN:
- **Convolución:** Desliza un filtro sobre la imagen
- **Pooling:** Reduce dimensionalidad, mantiene features importantes
- **Stride:** Cada cuántos píxeles se desliza el filtro
- **Padding:** Agrega bordes de ceros para no perder información

---

## 5️⃣ TRANSFORMERS (NIVEL BÁSICO)

### Estructura conceptual:
```
INPUT → Embedding → Positional Encoding → 
    [Self-Attention → Feed-Forward] × N →
OUTPUT
```

### Con Hugging Face (FÁCIL):
```python
from transformers import BertTokenizer, BertModel

# Cargar modelo preentrenado
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenizar texto
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors='pt')

# Forward pass
outputs = model(**inputs)
embeddings = outputs.last_hidden_state

# Para clasificación
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
```

### Key concepts:
- **Attention:** El modelo puede "enfocarse" en partes importantes del texto
- **Multi-head attention:** Múltiples perspectivas simultáneamente
- **Self-attention:** Cada palabra ve relación con todas las otras
- **Positional encoding:** Inyecta información del orden de las palabras

---

## 6️⃣ RNNs & LSTMs

```python
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch_size, sequence_length, hidden_size)
        
        # Usar último output
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output
```

### RNN vs LSTM:
- **RNN:** Simple, tiene problema de vanishing gradients
- **LSTM:** Usa gates (input, forget, output) para controlar flujo de información
- **GRU:** Versión simplificada de LSTM

---

## 7️⃣ TRANSFER LEARNING

```python
import torchvision.models as models

# Cargar modelo preentrenado (entrenado en ImageNet)
model = models.resnet18(pretrained=True)

# Opción 1: Feature extraction (congelar pesos)
for param in model.parameters():
    param.requires_grad = False

# Reemplazar última capa
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Opción 2: Fine-tuning (descongelar algunas capas)
for param in model.layer4.parameters():
    param.requires_grad = True

# Entrenar normalmente
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

---

## 8️⃣ PÉRDIDAS Y OPTIMIZADORES

### Loss functions:
```python
# Clasificación
nn.CrossEntropyLoss()      # Multiclase (softmax + log loss)
nn.BCELoss()               # Binaria

# Regresión
nn.MSELoss()               # Mean Squared Error
nn.L1Loss()                # Mean Absolute Error
```

### Optimizadores:
```python
optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
optim.RMSprop(model.parameters(), lr=0.001)

# Scheduler para ajustar learning rate
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler.step()  # Llamar cada epoch
```

---

## 9️⃣ DATA LOADERS (PyTorch)

```python
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# Crear dataset
dataset = TensorDataset(X_tensor, y_tensor)

# DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

# Usar en entrenamiento
for batch_X, batch_y in train_loader:
    output = model(batch_X)
    loss = loss_fn(output, batch_y)
    # ...

# Para imágenes (MNIST, CIFAR-10)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

---

## 🔟 GENERATIVE MODELS (Básico)

### Autoencoders:
```python
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 20)
        
        # Decoder
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        latent = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(latent))
        x = torch.sigmoid(self.fc4(x))
        return x
```

Uso:
```python
model = Autoencoder()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    for batch_X, _ in train_loader:
        reconstruction = model(batch_X)
        loss = loss_fn(reconstruction, batch_X)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 1️⃣1️⃣ EVALUACIÓN EN DEEP LEARNING

```python
# Evaluación en test set
model.eval()  # Modo evaluación (desactiva dropout, etc)
with torch.no_grad():
    y_pred = model(X_test)
    test_loss = loss_fn(y_pred, y_test)

# Métricas de clasificación
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred_class = y_pred.argmax(dim=1).numpy()
accuracy = accuracy_score(y_test.numpy(), y_pred_class)
f1 = f1_score(y_test.numpy(), y_pred_class, average='weighted')

print(f"Test Loss: {test_loss.item()}")
print(f"Accuracy: {accuracy}")
print(f"F1-Score: {f1}")

# Visualizar predicciones
import matplotlib.pyplot as plt
plt.imshow(X_test[0].reshape(28, 28))
plt.title(f"Predicted: {y_pred_class[0]}")
plt.show()
```

---

## 1️⃣2️⃣ TÉCNICAS DE REGULARIZACIÓN

```python
# Dropout (durante entrenamiento, 50% de neuronas se desactivan)
class RegularizedNN(nn.Module):
    def __init__(self):
        super(RegularizedNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout1(x)  # Solo activo durante entrenamiento
        
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x

# Batch Normalization
class BatchNormNN(nn.Module):
    def __init__(self):
        super(BatchNormNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
```

---

## 1️⃣3️⃣ HUGGING FACE (NLP RÁPIDO)

```python
from transformers import pipeline

# Clasificación de texto
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("This is a great movie!")  # {"label": "POSITIVE", "score": 0.99}

# Generación de texto
generator = pipeline("text-generation", model="gpt2")
result = generator("Once upon a time")

# Question answering
qa_pipeline = pipeline("question-answering")
context = "My name is Hugo. I live in Spain."
question = "Where does Hugo live?"
result = qa_pipeline(question=question, context=context)

# RAG (Retrieval Augmented Generation)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

inputs = tokenizer("Text to summarize...", return_tensors="pt")
summary_ids = model.generate(inputs["input_ids"])
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```

---

## 1️⃣4️⃣ DETECCIÓN DE OVERFITTING

```python
# Guardar métricas durante entrenamiento
train_losses = []
val_losses = []

for epoch in range(epochs):
    # Training
    train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
    train_losses.append(train_loss)
    
    # Validation
    val_loss = evaluate(model, val_loader, loss_fn)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")

# Graficar
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.show()

# Si train_loss << val_loss → OVERFITTING
```

---

## 1️⃣5️⃣ CHECKPOINTS (Guardar modelos)

```python
import torch

# Guardar
torch.save(model.state_dict(), 'model.pth')

# Cargar
model = SimpleCNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Guardar todo (modelo + optimizer)
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}, 'checkpoint.pth')

# Cargar
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

---

## 🎯 CHECKLIST PARA EJERCICIOS DE DL

- [ ] ¿Dataset está cargado y normalizado?
- [ ] ¿Datos divididos en train/val/test?
- [ ] ¿Arquitectura es apropiada para el problema?
- [ ] ¿Loss function es la correcta?
- [ ] ¿Optimizer está configurado?
- [ ] ¿Datos en formato correcto para el modelo?
- [ ] ¿Learning rate es razonable?
- [ ] ¿Entrenamiento converge?
- [ ] ¿Hay overfitting?
- [ ] ¿Métricas de evaluación son relevantes?

---

**Última actualización:** 12 Mayo 2026
**Para el examen:** 19 Mayo 2026

