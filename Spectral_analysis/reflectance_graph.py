# generate relectance vs wavelength graph
from osgeo import gdal
import matplotlib.pyplot as plt
import rasterio
import cv2
import numpy as np

# Input Type: Numpy Array
# Output Type: Plot


def plot_reflectance_graph(blue_ref, green_ref, red_ref, rededge_ref, nir_ref):
    band_list = ['Blue', 'Green', 'Red', 'RedEdge', 'NIR']

    blue_ref = blue_ref.flatten()
    green_ref = green_ref.flatten()
    red_ref = red_ref.flatten()
    rededge_ref = rededge_ref.flatten()
    nir_ref = nir_ref.flatten()

    a1, a2, x = np.min(blue_ref), np.max(blue_ref), np.mean(blue_ref)
    b1, b2, y = np.min(green_ref), np.max(green_ref), np.mean(green_ref)
    c1, c2, z = np.min(red_ref), np.max(red_ref), np.mean(red_ref)
    d1, d2, w = np.min(rededge_ref), np.max(rededge_ref), np.mean(rededge_ref)
    e1, e2, v = np.min(nir_ref), np.max(nir_ref), np.mean(nir_ref)

    # Wavelength Range for each spectra
    wavelength_ranges = ["450-500", "500-600", "600-700", "700-750", "750-900"]
    wavelength_midpoints = [(int(start) + int(end)) / 2 for start,
                            end in [range.split('-') for range in wavelength_ranges]]

    reflectance = [x, y, z, w, v]
    plt.plot(wavelength_midpoints, reflectance, marker='o', linestyle='-')
    plt.xlabel("Wavelength(nm)")
    plt.ylabel("Reflectance(nm)")
    plt.title("Reflectance vs Wavelength")
    plt.xticks(wavelength_midpoints, wavelength_ranges)
    plt.grid(True)
    plt.show()


def plot_min_max_graph(blue_ref, green_ref, red_ref, rededge_ref, nir_ref):
    band_list = ['Blue', 'Green', 'Red', 'RedEdge', 'NIR']
    band_wavelength_ranges = {'Blue': (450, 500), 'Green': (500, 600), 'Red': (600, 700),
                              'Red Edge': (700, 750), 'NIR': (750, 900)}

    blue_ref = blue_ref.flatten()
    green_ref = green_ref.flatten()
    red_ref = red_ref.flatten()
    rededge_ref = rededge_ref.flatten()
    nir_ref = nir_ref.flatten()

    a1, a2, x = np.min(blue_ref), np.max(blue_ref), np.mean(blue_ref)
    b1, b2, y = np.min(green_ref), np.max(green_ref), np.mean(green_ref)
    c1, c2, z = np.min(red_ref), np.max(red_ref), np.mean(red_ref)
    d1, d2, w = np.min(rededge_ref), np.max(rededge_ref), np.mean(rededge_ref)
    e1, e2, v = np.min(nir_ref), np.max(nir_ref), np.mean(nir_ref)

    num_images = 5

    band_indices = np.arange(len(band_list))

    for i in range(num_images):
        min_values = [a1, b1, c1, d1, e1]
        max_values = [a2, b2, c2, d2, e2]

        plt.plot(band_indices, min_values, marker='o',
                 linestyle='-', label=f'Min Values - Image {i + 1}')
        plt.plot(band_indices, max_values, marker='x',
                 linestyle='--', label=f'Max Values - Image {i + 1}')

    plt.xlabel('Wavelength Range (nm)')
    plt.ylabel('Pixel Value')
    plt.title('Min and Max Pixel Values for Each Band Across Images')
    plt.xticks(band_indices, [
               f"{band}\n({band_wavelength_ranges[band][0]}-{band_wavelength_ranges[band][1]} nm)" for band in band_list])
    plt.legend()
    plt.grid(True)
    plt.show()
