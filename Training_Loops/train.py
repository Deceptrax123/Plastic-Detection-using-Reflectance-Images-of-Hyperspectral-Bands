from sklearn.model_selection import train_test_split
import numpy as np
from dotenv import load_dotenv
from torch import mps, nn, multiprocessing
from torch.utils.data import DataLoader
import wandb
from Training_Loops.plastic_dataset import PlasticHyperspectralDataset
from Models.segnet import SegnetHyperSpectral
from Models.Unet.unet import UnetHyperSpectral
import os
import gc


def train_epoch():
    pass


def test_epoch():
    pass


def training_loop():
    pass


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
