import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Models.segnet import SegnetHyperSpectral
from skimage.transform import resize
from Models.hyper_cnn import HyperCNN
from Training_Loops.plastic_dataset import PlasticHyperspectalInference
from Training_Loops.plastic_dataset import PlasticHyperCNNInferenceDataset
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


def to_numpy(tens, shape):
    pred_img = np.transpose(tens, (0, 2, 3, 1))[0]

    blue_np, green_np, red_np, rededge_np, nir_np = pred_img[:, :, 0], pred_img[:, :, 1], \
        pred_img[:, :, 2], pred_img[:, :, 3], \
        pred_img[:, :, 4]
    blue_np, green_np, red_np, rededge_np, nir_np = np.reshape(blue_np, shape), \
        np.reshape(green_np, shape), \
        np.reshape(red_np, shape), \
        np.reshape(rededge_np, shape), \
        np.reshape(nir_np, shape)

    return blue_np, green_np, red_np, rededge_np, nir_np


def evaluate_spectral_graphs(shape):
    for step, (x, y) in enumerate(inference_loader):
        pred = model(x)
        pred_nograd = pred.detach().numpy()
        y = y.detach().numpy()

        pred_ref = to_numpy(pred_nograd, shape)
        y_ref = to_numpy(y, shape)

        compare_reflectance_graph(pred_ref, y_ref)


def inference(shape):
    for step, (x, y) in enumerate(inference_loader):
        pred = model(x)
        pred_nograd = pred.detach().numpy()
        y = y.detach().numpy()

        pred_img = np.transpose(pred_nograd, (0, 2, 3, 1))[0]

        blue_np, green_np, red_np, rededge_np, nir_np = pred_img[:, :, 0], pred_img[:, :, 1], \
            pred_img[:, :, 2], pred_img[:, :, 3], \
            pred_img[:, :, 4]
        blue_np, green_np, red_np, rededge_np, nir_np = np.reshape(blue_np, shape), \
            np.reshape(green_np, shape), \
            np.reshape(red_np, shape), \
            np.reshape(rededge_np, shape), \
            np.reshape(nir_np, shape)

        plt.imshow(green_np)
        plt.show()  # MAKE A SUBPLOT FIGURE HERE TO DISPLAY ALL BANDS

        break


if __name__ == '__main__':

    print("Enter Choice of Model")
    print("Segnet----->0")
    print("Hyperspectral CNN----->1")

    m = int(input())
    if m == 0:
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

        evaluate_spectral_graphs(shape=(1024, 1024, 1))
    elif m == 1:
        weights = torch.load(
            "Training_Loops/weights/hyper_cnn/model120.pth", map_location='cpu')

        model = HyperCNN()
        model.eval()

        model.load_state_dict(weights)

        dataset_paths = get_dataset()

        # Take a Random sample of paths
        dataset_paths_sampled = random.choices(dataset_paths, k=8)

        inference_set = PlasticHyperCNNInferenceDataset(dataset_paths_sampled)
        params = {
            'batch_size': 1,
            'num_workers': 0,
            'shuffle': True
        }

        inference_loader = DataLoader(inference_set, **params)

        print("Click 1 to compare Reflectances and click 2 to get reflectance images")

        ch = int(input())
        if ch == 1:
            evaluate_spectral_graphs(shape=(1300, 1600, 1))
        elif ch == 2:
            inference(shape=(1300, 1600, 1))
    else:
        print("Invalid..exit")
