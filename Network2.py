# Imports
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from customDataset import EdgeImages
from Architecture import (MyNet, block)
from torch.utils.data import DataLoader  # Gives easier dataset managment and creates mini batches
from tqdm import tqdm

# Hyperparameters
learning_rate = 1e-3
batch_size = 1
num_epochs = 10
layers = [3, 3, 3]
rollSets = [0, 0, 0]

# Load Data
total_testImages = 256
path = 'drive/MyDrive/EdgeImages'

dataset = EdgeImages(root_dir=path, transform=transforms.ToTensor())
train_set, test_set = torch.utils.data.random_split(dataset, [dataset.__len__() - total_testImages, total_testImages])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = MyNet(layers=layers, rollSets=rollSets)
model.to(device)

# Loss and optimizer
criterion = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    Mean = torch.zeros(1000, 1)
    Max = torch.zeros(1000, 1)

    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data = data.to(device=device)
            target = target.to(device=device)

            predictions = model(data)
            accuracy = (predictions - target) ** 2
            accuracy = torch.reshape(accuracy, (3, -1))
            accuracy = torch.sum(accuracy, 0)
            accuracy = torch.sqrt(accuracy)

            Mean[batch_idx] = torch.mean(accuracy)
            Max[batch_idx] = torch.max(accuracy)

        print(
            f"Mean {torch.mean(Mean)} max {torch.mean(Max)}"
        )

    model.train()


# Train Network
for epoch in range(num_epochs):
    losses = []
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for batch_idx, (data, target) in loop:
        # Get data to cuda if possible
        data = data.to(device=device)
        data = data.float()
        target = target.to(device=device)
        target = target.float()

        # forward
        output = model(data)
        loss = criterion(output, target)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

        # Update progress bar
        loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
        loop.set_postfix(loss=loss.item())

    print(f"Cost at epoch {epoch} is {sum(losses) / len(losses)}")
    print("Checking accuracy on Test Set")
    check_accuracy(test_loader, model)

# print("Checking accuracy on Training Set")
# check_accuracy(train_loader, model)

# print("Checking accuracy on Test Set")
# check_accuracy(test_loader, model)

