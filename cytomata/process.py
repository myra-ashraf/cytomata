import numpy as np
from scipy import ndimage as ndi
from skimage import img_as_float
from skimage.io import imread
from skimage.filters import (gaussian, laplace, median,
    threshold_li, threshold_yen, threshold_isodata, threshold_otsu)
from skimage.morphology import (remove_small_objects, remove_small_holes,
    disk, binary_erosion, binary_opening, erosion)
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt
import seaborn as sns
from cytomata.utils import custom_styles, custom_palette
from cytomata.GHT import threshold_GHT

def preprocess_img(imgf):
    """Subtract background and denoise fluorescence image."""
    img = img_as_float(imread(imgf))
    raw = img.copy()
    sig = estimate_sigma(img)
    bkg = img.copy()
    tval = threshold_li(bkg)
    broi = bkg*(bkg < tval)
    broi = broi[broi > 0]
    tval = np.percentile(broi, 25)
    bkg[bkg > tval] = tval
    bkg = gaussian(bkg, 50)
    img = img - bkg
    img[img < 0] = 0
    den = denoise_nl_means(img, h=sig, sigma=sig, patch_size=3, patch_distance=5)
    den = den - 2*sig
    den[den < 0] = 0
    return img, raw, bkg, den


def segment_object(img, rs=5000, fh=500, cb=None, er=11, factor=1):
    """Segment out bright objects from fluorescence image."""
    img = median(img)
    thv_iso = threshold_isodata(img) / 15
    thv_ots = threshold_otsu(img) / 15
    thv_yen = threshold_yen(img) / 15
    thv_li = threshold_li(img) / 4
    offset = (np.median(img[np.nonzero(img)])*2.25)**2
    thv = np.median([thv_iso, thv_ots, thv_yen, thv_li])*factor + offset
    thr = img > thv
    if er is not None:
        thr = binary_erosion(thr, selem=disk(11))
    if rs is not None:
        thr = remove_small_objects(ndi.label(thr)[0].astype(bool), min_size=rs)
    if fh is not None:
        thr = remove_small_holes(thr.astype(bool), area_threshold=fh)
    if cb is not None:
        thr = clear_border(thr, buffer_size=cb)
    thr = median(thr)
    return thr


def segment_clusters(img):
    """Segment out bright clusters from fluorescence image."""
    log = laplace(gaussian(img, sigma=1.5))
    thr = log > 0.1*np.std(img.ravel())
    return thr
