import numpy as np
from scipy import ndimage as ndi
from skimage import img_as_float
from skimage.io import imread
from skimage.filters import gaussian, laplace, median, threshold_li, threshold_triangle
from skimage.morphology import remove_small_objects, remove_small_holes, disk, binary_erosion, erosion
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.segmentation import clear_border
import time

def preprocess_img(imgf, tval=None):
    """Subtract background and denoise fluorescence image."""
    img = img_as_float(imread(imgf))
    sig = estimate_sigma(img)
    img = denoise_nl_means(img, h=sig, sigma=sig, patch_size=9, patch_distance=7)
    bkg = img.copy()
    if tval is None:
        tval = threshold_triangle(bkg)
    bkg[bkg > tval] = tval
    bkg = gaussian(bkg, 32)
    img = img - bkg
    img[img < 0] = 0
    return img, tval


def segment_object(img, rs=2000, fh=100, er=None, cb=None):
    """Segment out bright objects from fluorescence image."""
    thr = median(img > threshold_li(img))
    if rs is not None:
        thr = remove_small_objects(ndi.label(thr)[0].astype(bool), min_size=rs)
    if fh is not None:
        thr = remove_small_holes(thr.astype(bool), area_threshold=fh)
    if er is not None:
        thr = binary_erosion(thr, selem=disk(er))
    if cb is not None:
        thr = clear_border(thr, buffer_size=cb)
    return thr


def segment_clusters(img):
    """Segment out bright clusters from fluorescence image."""
    log = laplace(gaussian(img, sigma=1.5))
    thr = log > 0.1*np.std(img.ravel())
    return thr
