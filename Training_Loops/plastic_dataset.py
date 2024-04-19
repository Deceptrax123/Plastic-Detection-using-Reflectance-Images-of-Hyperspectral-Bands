import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import numpy as np
from dotenv import load_dotenv
from skimage.transform import resize
from Spectral_analysis.reflectance_image import get_spectral_bands, convert_to_reflectance
import os


class PlasticHyperspectralDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.calibration = [
            {"gain": 8.000, "offset": 1.309057, "band_name": "Blue"},
            {"gain": 8.000, "offset": 0.885130, "band_name": "Green"},
            {"gain": 4.500, "offset": 0.748694, "band_name": "Red"},
            {"gain": 4.50, "offset": 0.827855, "band_name": "Red edge"},
            {"gain": 5.00, "offset": 0.833119, "band_name": "NIR"}
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # get the paths of the 6 spectra
        spectra = self.paths[index]

        # Get X
        # Ignore the RGB band
        bands_x = get_spectral_bands(spectra[1:])
        bands_x = np.reshape(bands_x, (1300, 1600, 5))

        bands_x = resize(bands_x, (1024, 1024))

        transforms_x = T.Compose(
            [T.ToTensor(), T.Normalize(mean=(np.mean(bands_x[:, :, 0]), np.mean(bands_x[:, :, 1]),
                                       np.mean(bands_x[:, :, 2]), np.mean(bands_x[:, :, 3]), np.mean(bands_x[:, :, 4])),
                                       std=(np.std(bands_x[:, :, 0]), np.std(bands_x[:, :, 1]),
                                       np.std(bands_x[:, :, 2]), np.std(bands_x[:, :, 3]), np.std(bands_x[:, :, 4])))])
        transforms_y = T.Compose([
            T.ToTensor()])

        # Generate Y
        bands = list()
        for i, path in enumerate(spectra[1:]):
            reflectance_image = convert_to_reflectance(path, self.calibration[i]['gain'],
                                                       self.calibration[i]['offset'], self.calibration[i]['band_name'])
            bands.append(reflectance_image)

        # To convert to torch tensor since transpose takes place
        bands_arr = np.reshape(np.array(bands), (1300, 1600, 5))
        bands_arr = resize(bands_arr, (1024, 1024))

        # Convert to Tensors
        x_tensor = transforms_x(bands_x.astype(np.float32))
        y_tensor = transforms_y(bands_arr.astype(np.float32))

        return x_tensor, y_tensor
