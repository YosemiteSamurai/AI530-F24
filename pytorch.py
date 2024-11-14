# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

import torch
import time 
import array
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

timearr = []
labelarr = []
labelarr.append("Start")
timearr.append(time.time())

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

labelarr.append("Create")
timearr.append(time.time())

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

labelarr.append("Define")
timearr.append(time.time())

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

labelarr.append("Train")
timearr.append(time.time())

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

labelarr.append("Test")
timearr.append(time.time())
            
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    labelarr.append("Epoch {t+1}")
    timearr.append(time.time())
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

labelarr.append("End")
timearr.append(time.time())

print("Done!")
print (labelarr)
print (timearr)

# Start,Create,Define,Train,Test,Epoch1,Epoch2,Epoch3,Epoch4,Epoch5,End

# Arch1) This is the baseline architecture from the tutorial
# nn.Linear(28*28, 512),
# nn.ReLU(),
# nn.Linear(512, 512),
# nn.ReLU(),
# nn.Linear(512, 10)
# Accuracy: 65.4%, Avg loss: 1.084491
# 1731118452.0001903, 1731118452.0965602, 1731118452.1128042, 1731118452.1279354, 1731118452.1279354, 1731118452.1279354, 1731118481.547508, 1731118513.2514207, 1731118537.7830586, 1731118562.6830385, 1731118586.669984

# Arch2) The layers are Linear, ReLU, Linear, ReLU, Linear, ReLU, Linear (so we add a Linear+ReLU hidden layer to the base architecture). The layer sizes are same as in Arch1 (so first linear layer is 28*28-by-512, etc)
# nn.Linear(28*28, 512),
# nn.ReLU(),
# nn.Linear(512, 512),
# nn.ReLU(),
# nn.Linear(512, 512),
# nn.ReLU(),
# nn.Linear(512, 10)
# Accuracy: 48.3%, Avg loss: 1.629602
# 1731119197.377715, 1731119197.4849312, 1731119197.5008361, 1731119197.5280058, 1731119197.5280058, 1731119197.5280058, 1731119231.164022, 1731119266.3948152, 1731119301.6567428, 1731119335.3829563, 1731119366.9667103

# Arch3) The layers are Linear, ReLU, Linear (so we removed a Linear+ReLU hidden layer from Arch1). The layer sizes are same as in Arch1 (512 outputs in layer 1, etc)
# nn.Linear(28*28, 512),
# nn.ReLU(),
# nn.Linear(512, 10)
# Accuracy: 68.1%, Avg loss: 0.925611
# 1731119414.3985553, 1731119414.4948235, 1731119414.5189097, 1731119414.5279663, 1731119414.5279663, 1731119414.5279663, 1731119440.678027, 1731119464.318519, 1731119485.2804053, 1731119508.865268, 1731119532.161414

# Arch4) The layers are Linear(28*28, 512), ReLU, Linear(512, 1024), ReLU, Linear(1024, 512), ReLU, Linear(512,10). (So we added a hidden layer and increased the size of it)
# nn.Linear(28*28, 512),
# nn.ReLU(),
# nn.Linear(512, 1024),
# nn.ReLU(),
# nn.Linear(1024, 512),
# nn.ReLU(),
# nn.Linear(512, 10)
# Accuracy: 52.8%, Avg loss: 1.512417
# 1731119572.482335, 1731119572.5788262, 1731119572.5957468, 1731119572.644519, 1731119572.644519, 1731119572.644519, 1731119607.2557306, 1731119638.386417, 1731119673.7981157, 1731119710.748538, 1731119742.5495214