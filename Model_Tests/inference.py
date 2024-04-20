import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Models.segnet import SegnetHyperSpectral
from skimage.transform import resize
from Training_Loops.plastic_dataset import PlasticHyperspectalInference
from Spectral_analysis.reflectance_graph import plot_reflectance_graph
from Spectral_analysis.reflectance_graph import compare_reflectance_graph
import numpy as np
from dotenv import load_dotenv
import os
import random


def get_dataset():
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

    return dataset_paths


def to_numpy(tens):
    pred_img = np.transpose(tens, (0, 2, 3, 1))[0]

    blue_np, green_np, red_np, rededge_np, nir_np = pred_img[:, :, 0], pred_img[:, :, 1], \
        pred_img[:, :, 2], pred_img[:, :, 3], \
        pred_img[:, :, 4]
    blue_np, green_np, red_np, rededge_np, nir_np = np.reshape(blue_np, (1024, 1024, 1)), \
        np.reshape(green_np, (1024, 1024, 1)), \
        np.reshape(red_np, (1024, 1024, 1)), \
        np.reshape(rededge_np, (1024, 1024, 1)), \
        np.reshape(nir_np, (1024, 1024, 1))

    return blue_np, green_np, red_np, rededge_np, nir_np


def evaluate_spectral_graphs():
    for step, (x, y) in enumerate(inference_loader):
        pred = model(x)
        pred_nograd = pred.detach().numpy()
        y = y.detach().numpy()

        pred_ref = to_numpy(pred_nograd)
        y_ref = to_numpy(y)

        compare_reflectance_graph(pred_ref, y_ref)


def inference():
    for step, (x, y) in enumerate(inference_loader):
        pred = model(x)
        pred_nograd = pred.detach().numpy()
        y = y.detach().numpy()

        pred_img = np.transpose(pred_nograd, (0, 2, 3, 1))[0]

        blue_np, green_np, red_np, rededge_np, nir_np = pred_img[:, :, 0], pred_img[:, :, 1], \
            pred_img[:, :, 2], pred_img[:, :, 3], \
            pred_img[:, :, 4]
        blue_np, green_np, red_np, rededge_np, nir_np = np.reshape(blue_np, (1024, 1024, 1)), \
            np.reshape(green_np, (1024, 1024, 1)), \
            np.reshape(red_np, (1024, 1024, 1)), \
            np.reshape(rededge_np, (1024, 1024, 1)), \
            np.reshape(nir_np, (1024, 1024, 1))

        plot_reflectance_graph(blue_np, green_np, red_np, rededge_np, nir_np)


if __name__ == '__main__':
    weights = torch.load(
        "Training_Loops/weights/segnet/model260.pth", map_location='cpu')

    model = SegnetHyperSpectral()
    model.eval()

    model.load_state_dict(weights)

    dataset_paths = get_dataset()

    # Take a Random sample of paths
    dataset_paths_sampled = random.choices(dataset_paths, k=8)

    inference_set = PlasticHyperspectalInference(dataset_paths_sampled)
    params = {
        'batch_size': 1,
        'num_workers': 0,
        'shuffle': True
    }

    inference_loader = DataLoader(inference_set, **params)

    evaluate_spectral_graphs()
