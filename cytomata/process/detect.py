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
from skimage.filters import threshold_local, gaussian, laplace, threshold_otsu, sobel, median
from skimage.morphology import disk, binary_dilation, erosion, binary_erosion, remove_small_objects, watershed, binary_closing, binary_opening
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


def measure_regions(thr, img):
    """Measure the median intensity and areas of thresholded regions."""
    lab, _ = ndi.label(thr)
    rprops = regionprops(lab, img)
    areas = [prop.area for prop in rprops]
    int_imgs = [prop.intensity_image for prop in rprops]
    ave_ints = [prop.mean_intensity for prop in rprops]
    return np.array(areas), np.array(ave_ints)


def segment_cell(imgf, bkg_imgf=None, mask=None):
    """Segment out whole cell body from fluorescence images."""
    img, den = preprocess_img(imgf, bkg_imgf)
    sig = estimate_sigma(img)
    bkg = threshold_local(den, block_size=501, offset=sig, method='gaussian')
    thr = den > bkg
    if mask is not None:
        thr = thr | mask
    thr = binary_erosion(thr, selem=disk(25))
    thr = binary_dilation(thr, selem=disk(21))
    thr = clear_border(remove_small_objects(thr, min_size=10000), buffer_size=15)
    return thr, den


def segment_clusters(imgf, bkg_imgf=None, mask=None):
    """Segment out bright clusters from fluorescence images."""
    img, den = preprocess_img(imgf, bkg_imgf)
    edg = gaussian(laplace(den), sigma=1.25)
    sig = estimate_sigma(edg)
    bkg = threshold_local(edg, block_size=25, offset=-5e3*sig, method='gaussian')
    thr = edg > bkg
    if mask is not None:
        thr *= mask
    return thr, den


def segment_nucleus(imgf, bkg_imgf=None):
    """Segment out bright nucleus from dim cytoplasm."""
    img, den = preprocess_img(imgf, bkg_imgf)
    den = equalize_adapthist(den, clip_limit=0.02)
    bkg = threshold_local(den, block_size=501, offset=-0.1*np.max(den), method='gaussian')
    thr = den > bkg
    thr = binary_erosion(thr, selem=disk(9))
    nucl = clear_border(remove_small_objects(thr, min_size=1000), buffer_size=50)
    return nucl, den
