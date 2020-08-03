import os

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import img_as_float
from skimage.io import imread
from skimage.measure import label
from skimage.filters import (gaussian, laplace, median,
    threshold_li, threshold_yen, threshold_isodata, threshold_otsu)
from skimage.morphology import (remove_small_objects, remove_small_holes,
    disk, binary_erosion, binary_opening)
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.segmentation import clear_border

from cytomata.utils import setup_dirs


def preprocess_img(imgf):
    """Subtract background and denoise fluorescence image."""
    img = img_as_float(imread(imgf))
    raw = img.copy()
    bkg = img.copy()
    sig = estimate_sigma(img)
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
        thr = binary_erosion(thr, selem=disk(er))
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


def process_u_csv(ty, u_csv, save_dir):
    setup_dirs(save_dir)
    t_on = []
    udf = pd.read_csv(u_csv)
    ty = np.around(ty, 1)
    tu = np.around(np.arange(ty[0], ty[-1], 0.1), 1)
    uta = np.around(udf['ta'].values, 1)
    utb = np.around(udf['tb'].values, 1)
    u = np.zeros_like(tu)
    for ta, tb in zip(uta, utb):
        t_on += list(np.arange(round(ta, 1), round(tb, 1) + 0.01, 0.1))
        ia = list(tu).index(ta)
        ib = list(tu).index(tb)
        u[ia:ib+1] = 1
    t_ann_img = []
    for tbl in t_on:
        t_ann_img.append(min(ty, key=lambda ti : abs(ti - tbl)))
    u_data = np.column_stack((tu, u))
    u_path = os.path.join(save_dir, 'u.csv')
    np.savetxt(u_path, u_data, delimiter=',', header='t,u', comments='')
    return tu, u, t_ann_img