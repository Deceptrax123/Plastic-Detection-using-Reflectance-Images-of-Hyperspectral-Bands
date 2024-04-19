# compute radiance and reflectance images and find their maximum and minimum reflectances.
from osgeo import gdal
import matplotlib.pyplot as plt
import rasterio
import cv2
import numpy as np


def get_spectral_bands(file_paths):
    try:
        image_bands = list()  # 5X3
        for tif_file in file_paths:
            dataset = gdal.Open(tif_file)

            if dataset is None:
                print("Failed")
                return

            num_bands = dataset.RasterCount

            spectral_bands = []
            for band_index in range(1, num_bands+1):
                band = dataset.GetRasterBand(band_index)
                band_data = band.ReadAsArray()
                spectral_bands.append(band_data)  # dim=3

            image_bands.append(spectral_bands)

        return image_bands

    except Exception as e:
        print(f"Error: {str(e)}")


def convert_to_reflectance(image_path, gain, offset, band_name):
    panel_calibration = {
        "Blue": 0.67,
        "Green": 0.69,
        "Red": 0.68,
        "Red edge": 0.67,
        "NIR": 0.61
    }

    ulx, uly, lrx, lry = 799, 647, 840, 670

    with rasterio.open(image_path) as dataset:
        dn = dataset.read(1)  # Read from first band

    # Compute radiance and reflectance
    radiance = (dn*gain)+offset

    corrected_radiance = (gain*radiance)+offset

    radiance_image = np.reshape(corrected_radiance, dataset.shape)

    # Panel Region
    panel_region = radiance_image[uly:lry, ulx:lrx]

    mean_radiance = panel_region.mean()

    panel_reflectance = panel_calibration[band_name]

    # Get the reflectance image from radiance image
    radiance_to_reflectance = panel_reflectance/mean_radiance
    reflectance_image = radiance_image*radiance_to_reflectance

    return reflectance_image
