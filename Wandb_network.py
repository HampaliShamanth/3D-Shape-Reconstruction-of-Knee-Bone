import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from customDataset import EdgeImages
from Architecture import MyNet
from torch.utils.data import DataLoader  # Gives easier dataset managment and creates mini batches
from tqdm.notebook import tqdm

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2 ** 32 - 1)
np.random.seed(hash("improves reproducibility") % 2 ** 32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2 ** 32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2 ** 32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##
# %%capture
# !pip install wandb --upgrade

import wandb

wandb.login()

##
config = dict(
    epochs=10,
    layers=[2, 2, 2],
    rollSets=[8, 8, 8, 8],
    batch_size=16,
    learning_rate=0.001,
    dataset="Easy",
    architecture="CNN")


##


def model_pipeline(hyperparameters):
    # tell wandb to get started
    with wandb.init(project="pytorch-demo", config=hyperparameters):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        model, train_loader, test_loader, criterion, optimizer = make(config)

        # and use them to train the model
        train(model, train_loader, criterion, optimizer, config, test_loader)

        # and test its final performance
        test(model, test_loader)

    return model


##


def make(config):
    # Make the data
    total_testImages = 256

    if device.type == 'cpu':
        path = 'D:\OneDrive - University of Waterloo\Thesis\Projects\IC\Python\ML1\EdgeImages'
    else:
        # Running on cloud
        path = 'drive/MyDrive/EdgeImages'

    dataset = EdgeImages(root_dir=path, transform=transforms.ToTensor())
    train, test = torch.utils.data.random_split(dataset,
                                                [dataset.__len__() - total_testImages, total_testImages])

    train_loader = DataLoader(dataset=train, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=8)
    test_loader = DataLoader(dataset=test, batch_size=config.batch_size * 2, shuffle=True, pin_memory=True,
                             num_workers=8)

    # Make the model
    model = MyNet(layers=config.layers, rollSets=config.rollSets).to(device)

    # Make the loss and optimizer
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    return model, train_loader, test_loader, criterion, optimizer


##


def train(model, loader, criterion, optimizer, config, test_loader):
    # tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = len(loader) * config.epochs
    example_ct = 0  # number of examples seen

    for epoch in tqdm(range(config.epochs)):
        for _, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            data, target = data.float(), target.float()

            # Forward pass ➡
            output = model(data)
            loss = criterion(output, target)

            # Backward pass ⬅
            optimizer.zero_grad()
            loss.backward()

            # Step with optimizer
            optimizer.step()

            example_ct += len(data)

            # Report metrics at the end of the loop
            # if (example_ct + 1) == loader.__len__():
            #     train_log(loss, example_ct, epoch, output, target)
        train_log(loss, example_ct, epoch, output, target)
        test(model, test_loader)


def train_log(loss, example_ct, epoch, output, target):
    loss = float(loss)

    accuracy = (output - target) ** 2
    accuracy = torch.reshape(accuracy, (3, -1))
    accuracy = torch.sum(accuracy, 0)
    accuracy = torch.sqrt(accuracy)

    Mean = torch.mean(accuracy)
    Max = torch.max(accuracy)

    # where the magic happens
    wandb.log({"epoch": epoch, "loss": loss, "Mean_train": Mean, "Max_train": Max}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")



def test(model, test_loader):
    model.eval()
    Mean = torch.zeros(1000, 1)
    Max = torch.zeros(1000, 1)

    # Run the model on some test examples
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            data, target = data.float(), target.float()

            predictions = model(data)
            accuracy = (predictions - target) ** 2
            accuracy = torch.reshape(accuracy, (3, -1))
            accuracy = torch.sum(accuracy, 0)
            accuracy = torch.sqrt(accuracy)

            Mean[batch_idx] = torch.mean(accuracy)
            Max[batch_idx] = torch.max(accuracy)


        print(f"Mean {torch.mean(Mean)} max {torch.max(Max)}")

        wandb.log({"Mean_test": torch.mean(Mean), "Max_test": torch.max(Max)})

        # Save the model in the exchangeable ONNX format
        #torch.onnx.export(model, data, "model1.onnx")
        #wandb.save("model1.onnx")
    model.train()


# Build, train and analyze the model with the pipeline


model = model_pipeline(config)
