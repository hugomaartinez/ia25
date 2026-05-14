# CIFAR-10 Classification with PyTorch


## Introduction

The CIFAR-10 dataset is a collection of 60,000 32x32 color images, containing 10 object classes.

![cifar10](https://pytorch.org/tutorials/_static/img/cifar10.png)

## Environment Setup


```
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
```


```
np.random.seed(42)
torch.manual_seed(42)

# Determine the device (GPU if available, otherwise CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True # Choose the best algorithm according to hardware
    torch.cuda.manual_seed_all(42) # Initialize the seed for the GPU
else:
    device = torch.device("cpu")
    
print(f"\nUsing device: {device}")


num_workers = os.cpu_count() // 2 # Number of workers. Half of the available cores (common heuristic)
```

    
    Using device: cuda


## Data Loading and Preparation

We will use the [official PyTorch tutorial for CIFAR-10 classification](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) as a baseline model and seek to improve its performance.


```
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 'Lazy/Symmetric' Normalization with 0.5 for all channels
])

batch_size = 4

fulltrainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Separation into training and validation sets
train_size = int(0.8 * len(fulltrainset))
valid_size = len(fulltrainset) - train_size
trainset, validset = torch.utils.data.random_split(fulltrainset, [train_size, valid_size])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

fulltrainset.classes # Classes are already defined in the dataset
```

## Definition of Training and Validation Functions

There are different training and evaluation methodologies.
- Approach 1: Train completely, then evaluate (separate functions)
    - Simplicity: Simpler and easier to understand code.
    - Speed: Faster if you are only interested in the final result.
    - Less overhead: Does not interrupt the training process.
    - Suitable for large datasets: When evaluation is expensive.
- Approach 2: Evaluate after each epoch (integrated function)
    - Progress monitoring: You can see how the model improves during training.
    - Early detection of problems: Identify *overfitting/underfitting* in time.
    - *Early stopping*: You can stop training when it stops improving.
    - Saving the best model: Save the model with the best performance on validation.
    
In this case we use approach 2, which allows comparing different architectures by observing progress across epochs. Thus, we visualize not only the final results but also the learning curve.

Training and evaluation functions are defined so that they can be reused when training different architectures.


```
from utils.training_functions import train_model, evaluate_model, train_final_model
from utils.plot_functions import plot_metrics, plot_class_accuracy
```

## *Baseline*


```
# Initialize the baseline model
from models.baseline import Net
base_model = Net()

# Define loss criterion and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(base_model.parameters(), lr=0.001, momentum=0.9)

# Train the baseline model
base_metrics = train_model(
    base_model, 
    trainset,
    validset,
    batch_size,
    criterion,
    optimizer,
    epochs=15,
    device=device,
    num_workers=num_workers
)

plot_metrics([base_metrics], ['Base Model'])
```

    Starting training on 'cuda' for 15 epochs with batch size 32...


                                                                   

    Epoch 1/15 completed: Train Loss: 2.1475, Train Acc: 21.05%, Val Loss: 1.9132, Val Acc: 31.59%


                                                                   

    Epoch 2/15 completed: Train Loss: 1.7627, Train Acc: 36.24%, Val Loss: 1.6320, Val Acc: 41.59%


                                                                   

    Epoch 3/15 completed: Train Loss: 1.5546, Train Acc: 43.26%, Val Loss: 1.5017, Val Acc: 45.09%


                                                                   

    Epoch 4/15 completed: Train Loss: 1.4357, Train Acc: 48.00%, Val Loss: 1.4321, Val Acc: 48.05%


                                                                   

    Epoch 5/15 completed: Train Loss: 1.3602, Train Acc: 50.84%, Val Loss: 1.4035, Val Acc: 49.72%


                                                                  

    Epoch 6/15 completed: Train Loss: 1.3002, Train Acc: 53.23%, Val Loss: 1.3078, Val Acc: 53.20%


                                                                   

    Epoch 7/15 completed: Train Loss: 1.2433, Train Acc: 55.44%, Val Loss: 1.2818, Val Acc: 54.84%


                                                                  

    Epoch 8/15 completed: Train Loss: 1.1966, Train Acc: 57.30%, Val Loss: 1.2670, Val Acc: 55.72%


                                                                   

    Epoch 9/15 completed: Train Loss: 1.1610, Train Acc: 58.79%, Val Loss: 1.2064, Val Acc: 57.52%


                                                                  

    Epoch 10/15 completed: Train Loss: 1.1210, Train Acc: 60.05%, Val Loss: 1.2115, Val Acc: 57.21%


                                                                  

    Epoch 11/15 completed: Train Loss: 1.0891, Train Acc: 61.34%, Val Loss: 1.2015, Val Acc: 57.60%


                                                                  

    Epoch 12/15 completed: Train Loss: 1.0532, Train Acc: 62.51%, Val Loss: 1.1623, Val Acc: 59.42%


                                                                   

    Epoch 13/15 completed: Train Loss: 1.0222, Train Acc: 63.80%, Val Loss: 1.1456, Val Acc: 60.24%


                                                                  

    Epoch 14/15 completed: Train Loss: 0.9917, Train Acc: 65.01%, Val Loss: 1.1297, Val Acc: 61.12%


                                                                   

    Epoch 15/15 completed: Train Loss: 0.9684, Train Acc: 65.92%, Val Loss: 1.1324, Val Acc: 60.93%
    
    Training completed after finishing all epochs.
    Total training time: 27.27 seconds



    
![png](CIFAR-10_files/CIFAR-10_13_31.png)
    


We can observe *overfitting* as the training accuracy increases much more than the validation accuracy starting from around epoch 10.

## Improvement

To avoid *overfitting*, we can use regularization, dropout, etc.


```
from models.mejorado import ImprovedCNN
model2 = ImprovedCNN()

# Define loss criterion and optimizer
criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model2.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.AdamW(model2.parameters(), lr=0.001, weight_decay=1e-4)

# Train the model
metrics2 = train_model(
    model2, 
    trainset,
    validset,
    32,
    criterion,
    optimizer,
    epochs=15,
    device=device,
    num_workers=num_workers
)

### Visualization of metrics
plot_metrics([base_metrics, metrics2], ['Base Model', "Model2"])
```

    Starting training on 'cuda' for 15 epochs with batch size 32...


                                                                  

    Epoch 1/15 completed: Train Loss: 1.4799, Train Acc: 46.02%, Val Loss: 1.0774, Val Acc: 61.39%


                                                                  

    Epoch 2/15 completed: Train Loss: 1.1240, Train Acc: 60.22%, Val Loss: 0.9531, Val Acc: 65.72%


                                                                  

    Epoch 3/15 completed: Train Loss: 0.9633, Train Acc: 66.55%, Val Loss: 0.7946, Val Acc: 72.08%


                                                                  

    Epoch 4/15 completed: Train Loss: 0.8563, Train Acc: 70.33%, Val Loss: 0.7513, Val Acc: 74.20%


                                                                  

    Epoch 5/15 completed: Train Loss: 0.7667, Train Acc: 73.20%, Val Loss: 0.6711, Val Acc: 76.42%


                                                                  

    Epoch 6/15 completed: Train Loss: 0.7052, Train Acc: 75.68%, Val Loss: 0.6634, Val Acc: 77.70%


                                                                  

    Epoch 7/15 completed: Train Loss: 0.6414, Train Acc: 78.13%, Val Loss: 0.6576, Val Acc: 78.15%


                                                                  

    Epoch 8/15 completed: Train Loss: 0.5930, Train Acc: 79.55%, Val Loss: 0.6210, Val Acc: 79.19%


                                                                  

    Epoch 9/15 completed: Train Loss: 0.5492, Train Acc: 81.08%, Val Loss: 0.5852, Val Acc: 80.62%


                                                                  

    Epoch 10/15 completed: Train Loss: 0.5057, Train Acc: 82.43%, Val Loss: 0.5626, Val Acc: 81.58%


                                                                  

    Epoch 11/15 completed: Train Loss: 0.4714, Train Acc: 83.83%, Val Loss: 0.5843, Val Acc: 81.12%


                                                                  

    Epoch 12/15 completed: Train Loss: 0.4368, Train Acc: 84.79%, Val Loss: 0.5720, Val Acc: 81.86%


                                                                  

    Epoch 13/15 completed: Train Loss: 0.4074, Train Acc: 85.92%, Val Loss: 0.5657, Val Acc: 81.91%


                                                                  

    Epoch 14/15 completed: Train Loss: 0.3752, Train Acc: 87.05%, Val Loss: 0.5568, Val Acc: 82.68%


                                                                  

    Epoch 15/15 completed: Train Loss: 0.3545, Train Acc: 87.58%, Val Loss: 0.5935, Val Acc: 82.38%
    
    Training completed after finishing all epochs.
    Total training time: 39.60 seconds



    
![png](CIFAR-10_files/CIFAR-10_17_31.png)
    


## Test Evaluation and Storing the Final Model

Having chosen the newly proposed architecture as the final model, we retrain it with the entire training set.


```
model2 = ImprovedCNN() # Reinitialize the model (although there could be other strategies)
optimizer = torch.optim.AdamW(model2.parameters(), lr=0.001, weight_decay=1e-4)

final_metrics = train_final_model(
    model2,
    fulltrainset,
    batch_size,
    criterion,
    optimizer,
    epochs=10, # From here on we saw that overfitting began to occur
    device=device,
    num_workers=num_workers,
)
```

    Starting final training on 'cuda' for 10 epochs with batch size 32...


                                                                  

    Epoch 1/10 => Train Loss: 1.4207, Train Acc: 48.2580


                                                                  

    Epoch 2/10 => Train Loss: 1.0400, Train Acc: 63.3380


                                                                  

    Epoch 3/10 => Train Loss: 0.8959, Train Acc: 68.9160


                                                                  

    Epoch 4/10 => Train Loss: 0.7821, Train Acc: 73.1600


                                                                  

    Epoch 5/10 => Train Loss: 0.7082, Train Acc: 75.9800


                                                                  

    Epoch 6/10 => Train Loss: 0.6372, Train Acc: 78.0140


                                                                  

    Epoch 7/10 => Train Loss: 0.5846, Train Acc: 80.2040


                                                                  

    Epoch 8/10 => Train Loss: 0.5336, Train Acc: 81.7960


                                                                  

    Epoch 9/10 => Train Loss: 0.4920, Train Acc: 83.0180


                                                                  

    Epoch 10/10 => Train Loss: 0.4599, Train Acc: 84.1880
    
    Final training completed.
    Total final training time: 27.68 seconds


    


```
# Evaluate the final model on the test set
test_results = evaluate_model(model2, testloader, criterion, device) # Capture the full results
test_loss = test_results['loss']
test_acc = test_results['accuracy']
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

# Now plot the class accuracy using the captured results
plot_class_accuracy(
    results_list=[test_results],  # Pass the results dictionary in a list
    model_names=['Final Model'],  # Provide a name for the model
    class_names=testset.classes
)
```

                                                                  

    Test Loss: 0.5603, Test Accuracy: 81.75%


    


    
![png](CIFAR-10_files/CIFAR-10_21_3.png)
    


Once the model is chosen and trained, we can store it for deployment.


```
PATH = './cifar_net.pth'
torch.save(model2.state_dict(), PATH)
```

## Deploy Model to Hugging Face Hub

Finally, we can automate the uploading of the trained model to the Hugging Face Model Hub. This allows our FastAPI application (running in Hugging Face Spaces) to download the latest model weights dynamically.

> **Authentication Requirement**: 
> To upload models to Hugging Face, you need an Access Token with **WRITE** permissions. 
> You have two secure options to provide this token:
> 1. **`.env` file**: Create a `.env` file in the base folder and add `HF_TOKEN=your_token_here`. Then restart your environment, or just export it in your shell (`export HF_TOKEN=...`).
> 2. **Interactive Login**: Use `huggingface_hub.notebook_login()` to paste your token interactively.


```
import os
from huggingface_hub import HfApi, notebook_login

# Look for HF_TOKEN in environment variables
hf_token = os.environ.get("HF_TOKEN")

if not hf_token:
    print("HF_TOKEN not found in environment variables. Please login interactively:")
    notebook_login()
else:
    print("HF_TOKEN found in environment variables!")
```

    HF_TOKEN not found in environment variables. Please login interactively:



```
api = HfApi()

repo_id = "avidaldo/cifar-10-fastapi-model" # Target Model repo on Hugging Face

# Create the model repository if it doesn't exist
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

# Upload the model file
print(f"Uploading ./cifar_net.pth to {repo_id}...")
api.upload_file(
    path_or_fileobj='./cifar_net.pth',
    path_in_repo="cifar_net.pth",
    repo_id=repo_id,
    repo_type="model"
)

print(f"Model successfully uploaded to https://huggingface.co/{repo_id}")
```

    Uploading ./cifar_net.pth to avidaldo/cifar-10-fastapi-model...



    Processing Files (0 / 0): |          |  0.00B /  0.00B            



    New Data Upload: |          |  0.00B /  0.00B            


    Model successfully uploaded to https://huggingface.co/avidaldo/cifar-10-fastapi-model

