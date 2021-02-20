import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from customDataset import EdgeImages
from Architecture import MyNet
from torch.utils.data import DataLoader  # Gives easier dataset managment and creates mini batches
from tqdm.notebook import tqdm
import plotly.graph_objects as go
import os

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
    epochs=1,
    layers=[2, 2, 2],
    rollSets=[16, 0, 8, 0],
    batch_size=16,
    learning_rate=0.02,
    dataset="Easy",
    architecture="CNN", )

if device.type != 'cuda':
    config['epochs'] = 1
    config['batch_size'] = 1


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
        test(model, test_loader, export=True)

    return model


##


def make(config):
    # Make the data
    total_testImages = 256

    if device.type == 'cuda':
        # Running on cloud
        path = 'drive/MyDrive/EdgeImages'
    else:
        path = 'D:\OneDrive - University of Waterloo\Thesis\Projects\IC\Python\ML1\EdgeImages'

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
    example_ct = 0  # number of examples seen
    loop = tqdm(enumerate(loader), total=len(loader), leave=False)
    for epoch in range(config.epochs):
        for _, (data, target) in loop:
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

            # Update progress bar
            loop.set_description(f"Epoch [{epoch}/{config.epochs}]")
            loop.set_postfix(loss=loss.item())

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


def test(model, test_loader, export=False):
    model.eval()
    Mean = torch.zeros(1000, 1)
    Max = torch.zeros(1000, 1)
    pred_accumulator = torch.tensor([]).to(device)
    data_accumulator = torch.tensor([]).to(device)
    target_accumulator = torch.tensor([]).to(device)

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
            if export:
                pred_accumulator = torch.cat([pred_accumulator, predictions], dim=0)
                data_accumulator = torch.cat([data_accumulator, data], dim=0)
                target_accumulator = torch.cat([target_accumulator, target], dim=0)

        print(f"Mean {torch.mean(Mean)} max {torch.max(Max)}")

        wandb.log({"Mean_test": torch.mean(Mean), "Max_test": torch.max(Max)})

        # Save the model in the exchangeable ONNX format
        if export:
            import pickle

            with open('triangulation', 'rb') as f:
                triangulation = pickle.load(f)

            LogWandbData(pred_accumulator, target_accumulator, objType='Object3D', allFaces=triangulation)

            LogWandbData(data_accumulator, objType='Image')
            torch.onnx.export(model, data, "model1.onnx", opset_version=12)

            wandb.save("model1.onnx")

    model.train()


def LogWandbData(data, pred=None, objType='Image', n=50, allFaces=None):
    if objType == 'Object3D':
        target = data[0:n].to('cpu').numpy()
        pred = pred[0:n].to('cpu').numpy()

        target = target.reshape(target.shape[0], 3, -1)
        pred = pred.reshape(pred.shape[0], 3, -1)

        tar_pred = np.concatenate([target, pred], axis=1)
        data_list = map(list, tar_pred)

        def Createfig(N, allFaces):
            N = np.array([N]).squeeze()
            N = N[:, 0:3957]
            i, j, k = allFaces.T

            x, y, z = N[0:3, :]
            data1 = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='lightpink', opacity=0.50)
            x, y, z = N[3:6, :]
            data2 = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='red', opacity=0.50)
            # data3 = go.Scatter3d(x=x, y=y, z=z, mode="lines")

            fig = go.Figure()
            fig.add_trace(data1)
            fig.add_trace(data2)
            # fig.add_trace(data3)
            fig = fig.to_json()
            return fig

        dict = {str(count): [Createfig([N], allFaces)] for count, N in enumerate(data_list)}
        # pdb.set_trace()
        #data_list = map(list, data)

    if objType == 'Image':
        data = data[0:n].to('cpu').numpy()
        data_list = map(list, data)
        dict = {str(count): [wandb.Image(np.array([k]))] for count, k in enumerate(data_list)}
        

    wandb.log(dict)


#model = model_pipeline(config)
wandb.init(project="visualize-predictions", name="metrics")
# A=torch.randn(50,1024,1024)
# LogWandbData(A, objType='Image')

pred_accumulator=torch.randn(50,12576)
target_accumulator=torch.randn(50,12576)

import pickle
with open('drive/MyDrive/Code/triangulation', 'rb') as f:
    triangulation = pickle.load(f)
LogWandbData(pred_accumulator, target_accumulator, objType='Object3D', allFaces=triangulation)

wandb.finish()