import os

import numpy as np
from scipy import ndimage as ndi
from skimage import img_as_float
from skimage.measure import regionprops
from skimage.feature import peak_local_max
from skimage.restoration import denoise_nl_means
from skimage.filters import threshold_local, gaussian
from skimage.morphology import disk, dilation, remove_small_objects
from skimage.segmentation import random_walker, clear_border, find_boundaries

from cytomata.utils.visual import imshow


def extract_intensity(img):
    den = denoise_nl_means(img_as_float(img), h=0.005, multichannel=False)
    gau = gaussian(den, sigma=30)
    sub = den - gau
    sub[sub < 0] = 0
    return np.mean(sub[sub.nonzero()])


def extract_regions(img, denoise_h=0.005, gau_sigma=45, thres_block=55,
    thres_offset=0, peaks_min_dist=20, min_size=100, save_dir=None):
    den = denoise_nl_means(img_as_float(img), h=denoise_h, multichannel=False)
    ga0 = gaussian(den, sigma=gau_sigma)
    sub = den - ga0
    sub[sub < 0] = 0
    th = threshold_local(sub, block_size=thres_block, offset=thres_offset)
    ga1 = gaussian(sub, sigma=1)
    thres = ga1 > th
    labels, _ = ndi.label(thres)
    dist = ndi.distance_transform_edt(sub)
    lmax = peak_local_max(image=dist, labels=labels,
        min_distance=peaks_min_dist, indices=False, exclude_border=False)
    markers, n = ndi.label(dilation(lmax, disk(3)))
    markers[~thres] = -1
    rw = random_walker(sub, markers, beta=100, mode='bf')
    rw[rw < 0] = 0
    regions = clear_border(remove_small_objects(rw, min_size=min_size), buffer_size=3)
    bouns = find_boundaries(regions)
    final = sub.copy()
    final[bouns] = np.percentile(final, 99.99)
    if save_dir is not None:
        res_names = ['original', 'denoised', 'subtracted', 'thresholded', 'peaks', 'regions']
        res_imgs = [img, den, sub, thres, markers, final]
        for i, (res_name, res_img) in enumerate(zip(res_names, res_imgs)):
            res_dir = os.path.join(save_dir, str(i) + '_' + res_name)
            if not os.path.exists(res_dir):
                os.makedirs(res_dir)
            imshow(res_img, res_name.title(), os.path.join(res_dir, str(len(os.listdir(res_dir)))))
    return regions
