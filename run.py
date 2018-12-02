import os
import sys
import eel
import warnings
from base64 import b64encode

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from skimage.exposure import equalize_adapthist


from cytomata.utils.io import list_fnames, setup_dirs
from cytomata.process.extract import images_to_median_frame_intensities


eel.init('gui')


@eel.expose
def guess_img_dir():
    return os.path.join(os.path.expanduser('~'), 'Documents', 'imgs')

@eel.expose
def guess_out_dir():
    return os.path.join(os.path.expanduser('~'), 'Documents', 'output')

@eel.expose
def check_found_imgs(img_dir):
    try:
        return len(list_fnames(img_dir))
    except:
        return 0

def img_to_b64(img, touchup=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if touchup:
            equ = equalize_adapthist(img, clip_limit=0.005)
            cv_img = img_as_ubyte(equ)
            colored = cv2.applyColorMap(cv_img, cv2.COLORMAP_JET)
            return b64encode(cv2.imencode('.png', colored)[1]).decode('utf-8')
        else:
            return b64encode(cv2.imencode('.png', img_as_ubyte(img))[1]).decode('utf-8')

@eel.expose
def process_imgs(img_dir, out_dir, proto, params):
    setup_dirs(out_dir)
    if proto == '0':
        def iter_cb(med_int, imgs, prog):
            img0 = img_to_b64(imgs[0], touchup=True)
            img1 = img_to_b64(imgs[1], touchup=True)
            img2 = img_to_b64(imgs[2], touchup=True)
            img3 = img_to_b64(imgs[3], touchup=True)
            img4 = img_to_b64(imgs[4], touchup=False)
            eel.update_img_results(img0, img1, img2, img3, img4, med_int, prog)
            return eel.is_proc_imgs_stopped()()
        images_to_median_frame_intensities(
            img_dir, out_dir, int(params['gauss']), iter_cb
        )

options = {
    'mode': "chrome",
    'host': 'localhost',
    'port': 8080,
    'chromeFlags': ["--start-fullscreen"]
}

eel.start('index.html', options=options)
