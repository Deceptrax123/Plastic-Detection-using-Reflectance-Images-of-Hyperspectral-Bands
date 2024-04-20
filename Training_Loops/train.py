from sklearn.model_selection import train_test_split
import numpy as np
from dotenv import load_dotenv
import torch
from torch import mps, nn, multiprocessing
from torch.utils.data import DataLoader
import wandb
from Training_Loops.plastic_dataset import PlasticHyperspectralDataset
from Models.segnet import SegnetHyperSpectral
from Models.hyper_cnn import HyperCNN
from Models.initialize import initialize
import os
import gc


def train_epoch():
    epoch_loss = 0

    for step, (x_sample, y_sample) in enumerate(train_loader):
        x_sample = x_sample.to(device=device)
        y_sample = y_sample.to(device=device)

        predictions = model(x_sample)
        model.zero_grad()

        loss = objective(predictions, y_sample)

        # Backpropagation
        loss.backward()
        model_optimizer.step()

        epoch_loss += loss.item()

        # Memory management
        del x_sample
        del y_sample
        del predictions
        mps.empty_cache()
        gc.collect(generation=2)

    loss = epoch_loss/(step+1)

    return loss


def test_epoch():
    epoch_loss = 0

    for step, (x_sample, y_sample) in enumerate(test_loader):
        x_sample = x_sample.to(device=device)
        y_sample = y_sample.to(device=device)

        predictions = model(x_sample)
        loss = objective(predictions, y_sample)

        epoch_loss += loss.item()

        del x_sample
        del y_sample
        del predictions

        mps.empty_cache()
        gc.collect(generation=2)

    loss = epoch_loss/(step+1)
    return loss


def training_loop():
    for epoch in range(NUM_EPOCHS):
        model.train(True)

        train_loss = train_epoch()
        model.eval()

        with torch.no_grad():
            test_loss = test_epoch()
            print(f"Epoch: {epoch+1}")
            print(f"Train Loss: {train_loss}")
            print(f"Test Loss: {test_loss}")

            wandb.log({
                "Train Loss": train_loss,
                "Test Loss": test_loss,
                "Learning Rate": model_optimizer.param_groups[0]['lr']
            })

            # checkpoints
            if (epoch+1) % 10 == 0:
                path = f"Training_Loops/weights/hyper_cnn/model{epoch+1}.pth"
                torch.save(model.state_dict(), path)

            scheduler.step()  # Update learning rate


if __name__ == '__main__':
    multiprocessing.set_sharing_strategy('file_system')
    load_dotenv('.env')

    global_path = os.getenv("global")
    imgs = sorted(os.listdir(global_path))

    dataset_paths = list()

    history = list()
    for c, path in enumerate(imgs):
        if (c+1) % 6 != 0:
            history.append(path)
        else:
            history.append(path)
            dataset_paths.append(history)
            history = []  # empty history

    params = {
        'batch_size': 8,
        'shuffle': True,
        'num_workers': 0
    }

    train, test = train_test_split(dataset_paths, test_size=0.20)

    train_set = PlasticHyperspectralDataset(train)
    test_set = PlasticHyperspectralDataset(test)

    wandb.init(
        project="Plastic-Reflectance",
        config={
            'Dataset': "Self Accquired"
        }
    )

    train_loader = DataLoader(train_set, **params)
    test_loader = DataLoader(test_set, **params)

    device = torch.device("mps")

    LR = 0.01
    NUM_EPOCHS = 100000

    objective = nn.MSELoss()
    model = HyperCNN().to(device=device)
    model.apply(initialize)

    model_optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, betas=(0.9, 0.999))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        model_optimizer, 20, eta_min=0.0000000001, verbose=True)

    train_steps = (len(train)+params['batch_size']-1)//params['batch_size']
    test_steps = (len(test)+params['batch_size']-1)//params['batch_size']

    mps.empty_cache()
    gc.collect(generation=2)

    training_loop()
