# MNIST Classification with Convolutional Network

MNIST is a classic example of handwritten digit recognition. The MNIST database is used, containing 60,000 training images and 10,000 test images. Each image is 28x28 pixels and each pixel has a value between 0 and 255.

This dataset marked a milestone in the history of AI, with which [in 1998 Yann LeCun's team used a convolutional neural network to achieve a 0.8% error rate in digit recognition](https://www.youtube.com/watch?v=H0oEr40YhrQ), using the LeNet-5 architecture.

It is the same example used to explain [neural network theory in the 3Brown1Blue video](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi).


```
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import datasets, transforms
```

## Dataset Loading

We will often use more than one transformation to preprocess the data. For example, in the case of images, they are often normalized and resized. To do this efficiently, we can use the `Compose` class from `torchvision.transforms`.


```
# Define transformations for image preprocessing
# We use pre-computed standard values for MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
```


```
# Load MNIST dataset with normalization
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Create dataloaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
```

## Model Definition


```
from torch.nn import functional as F

class CNN(nn.Module): # Define the convolutional neural network
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) 
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.fc1 = nn.Linear(7 * 7 * 64, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 7 * 7 * 64) # Flatten after convolutional layers
    x = F.relu(self.fc1(x))
    x = F.log_softmax(self.fc2(x), dim=1)  # Use log_softmax for cross-entropy loss
    return x
  
model = CNN() # Instantiate the neural network
```

## Model Training

### Defining Loss Function and Optimizer

We define the loss function and the optimizer. In this case we will use the `optim.Adam` optimizer. Adam is a variant of stochastic gradient descent that calculates individual learning rates for different parameters.


```
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
```

### Training


```
model.train() # Set model to training mode (default behavior, but good practice)

for epoch in range(5): # Define 5 epochs
  
  for i, (images, labels) in enumerate(train_loader):
    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % 100 == 0:
      print(f'Epoch [{epoch+1}/{10}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
```

    Epoch [1/10], Step [100/938], Loss: 0.3468
    Epoch [1/10], Step [200/938], Loss: 0.1342
    Epoch [1/10], Step [300/938], Loss: 0.0703
    Epoch [1/10], Step [400/938], Loss: 0.0912
    Epoch [1/10], Step [500/938], Loss: 0.0450
    Epoch [1/10], Step [600/938], Loss: 0.0329
    Epoch [1/10], Step [700/938], Loss: 0.0649
    Epoch [1/10], Step [800/938], Loss: 0.0311
    Epoch [1/10], Step [900/938], Loss: 0.0800
    Epoch [2/10], Step [100/938], Loss: 0.0359
    Epoch [2/10], Step [200/938], Loss: 0.0082
    Epoch [2/10], Step [300/938], Loss: 0.0427
    Epoch [2/10], Step [400/938], Loss: 0.0304
    Epoch [2/10], Step [500/938], Loss: 0.0065
    Epoch [2/10], Step [600/938], Loss: 0.0244
    Epoch [2/10], Step [700/938], Loss: 0.0178
    Epoch [2/10], Step [800/938], Loss: 0.0107
    Epoch [2/10], Step [900/938], Loss: 0.0111
    Epoch [3/10], Step [100/938], Loss: 0.0017
    Epoch [3/10], Step [200/938], Loss: 0.0594
    Epoch [3/10], Step [300/938], Loss: 0.0644
    Epoch [3/10], Step [400/938], Loss: 0.0890
    Epoch [3/10], Step [500/938], Loss: 0.0044
    Epoch [3/10], Step [600/938], Loss: 0.0171
    Epoch [3/10], Step [700/938], Loss: 0.0018
    Epoch [3/10], Step [800/938], Loss: 0.0090
    Epoch [3/10], Step [900/938], Loss: 0.0074
    Epoch [4/10], Step [100/938], Loss: 0.0044
    Epoch [4/10], Step [200/938], Loss: 0.0029
    Epoch [4/10], Step [300/938], Loss: 0.0431
    Epoch [4/10], Step [400/938], Loss: 0.0540
    Epoch [4/10], Step [500/938], Loss: 0.0198
    Epoch [4/10], Step [600/938], Loss: 0.0160
    Epoch [4/10], Step [700/938], Loss: 0.0578
    Epoch [4/10], Step [800/938], Loss: 0.0262
    Epoch [4/10], Step [900/938], Loss: 0.0008
    Epoch [5/10], Step [100/938], Loss: 0.0106
    Epoch [5/10], Step [200/938], Loss: 0.0426
    Epoch [5/10], Step [300/938], Loss: 0.0081
    Epoch [5/10], Step [400/938], Loss: 0.0137
    Epoch [5/10], Step [500/938], Loss: 0.0006
    Epoch [5/10], Step [600/938], Loss: 0.0002
    Epoch [5/10], Step [700/938], Loss: 0.0154
    Epoch [5/10], Step [800/938], Loss: 0.0067
    Epoch [5/10], Step [900/938], Loss: 0.0144


## Model Evaluation


```
with torch.no_grad():
  correct = 0
  total = 0
  for images, labels in test_loader:
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
  print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
```

    Accuracy of the network on the 10000 test images: 99.12%


- https://dudeperf3ct.github.io/cnn/mnist/2018/10/17/Force-of-Convolutional-Neural-Networks/
