import time

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.stats import iqr
from skimage import img_as_float
from skimage.io import imread
from skimage.exposure import equalize_adapthist
from skimage.measure import regionprops
from skimage.feature import peak_local_max
from skimage.filters import threshold_local, gaussian, laplace, threshold_otsu, sobel, median, threshold_li
from skimage.morphology import disk, binary_dilation, erosion, binary_erosion, remove_small_objects, remove_small_holes, binary_closing, binary_opening
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.segmentation import random_walker, clear_border, find_boundaries

from cytomata.utils.io import list_img_files, setup_dirs
from cytomata.utils.visual import plot, custom_styles, custom_palette


def preprocess_img(imgf, bkg_imgf=None):
    """Clean up fluorescence images. Subtract background, adjust contrast, and denoise."""
    img = img_as_float(imread(imgf))
    if bkg_imgf:
        bkg_img = img_as_float(imread(bkg_imgf))
        img = img - bkg_img
    sig = estimate_sigma(img)
    den = denoise_nl_means(img, h=sig, sigma=sig, patch_size=5, patch_distance=7)
    return img, den


def subtract_img(imgf):
    img, den = preprocess_img(imgf)
    sig = estimate_sigma(img)
    # thr = median(den > threshold_li(den))
    bkg = threshold_local(den, block_size=701, offset=-sig, method='gaussian')
    # plt.plot(img[280,:])
    # plt.plot(bkg[280,:])
    # plt.show()
    sub = img - bkg
    sub[sub < 0] = 0
    return sub


def measure_regions(thr, img):
    """Measure the ave intensity and areas of thresholded regions."""
    lab, _ = ndi.label(thr)
    rprops = regionprops(lab, img)
    areas = [prop.area for prop in rprops]
    int_imgs = [prop.intensity_image for prop in rprops]
    ave_ints = [prop.mean_intensity for prop in rprops]
    return np.array(areas), np.array(ave_ints)


def segment_cell(imgf, bkg_imgf=None):
    """Segment out whole cell body from fluorescence images."""
    img, den = preprocess_img(imgf, bkg_imgf)
    thr = median(den > threshold_li(den))
    thr = remove_small_objects(ndi.label(thr)[0], min_size=5000)
    return thr, den


def segment_clusters(imgf, bkg_imgf=None):
    """Segment out bright clusters from fluorescence images."""
    img, den = preprocess_img(imgf, bkg_imgf)
    sig = estimate_sigma(img)
    log = laplace(gaussian(median(den), sigma=2))
    bkg = threshold_local(log, block_size=25, offset=-sig, method='gaussian')
    thr = log > bkg
    return thr, den


def segment_nucleus(imgf, bkg_imgf=None):
    """Segment out bright nucleus from dim cytoplasm."""
    img, den = preprocess_img(imgf, bkg_imgf)
    sig = estimate_sigma(img)
    bkg = threshold_local(den, block_size=501, offset=-sig, method='gaussian')
    thr = median(den > bkg)
    thr = remove_small_objects(ndi.label(thr)[0], min_size=500)
    thr = remove_small_holes(thr, area_threshold=3000)
    thr = clear_border(thr, buffer_size=1)
    return thr, den


def segment_dim_nucleus(imgf, bkg_imgf=None):
    """Segment out dim nucleus from bright cytoplasm."""
    img, den = preprocess_img(imgf, bkg_imgf)
    gos = gaussian(sobel(den), sigma=9)
    thr = gos > threshold_otsu(gos)
    thr = clear_border(remove_small_objects(thr, min_size=1000))
    cyto = erosion(thr, selem=disk(12))
    cell = ndi.binary_fill_holes(cyto)
    nucl = cell^cyto
    return nucl, den
