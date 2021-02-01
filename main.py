import os
import sys
import eel
import warnings
from base64 import b64encode

import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.exposure import equalize_adapthist

from cytomata.utils import list_img_files, setup_dirs
from scripts.process_imgs import process_fluo_timelapse, process_10x_imgs


@eel.expose
def guess_img_dir():
    return os.path.join(os.path.expanduser('~'), 'Documents', 'imgs')

@eel.expose
def guess_save_dir():
    return os.path.join(os.path.expanduser('~'), 'Documents', 'results')

@eel.expose
def check_dir_exists(dir):
    return os.path.exists(dir)

@eel.expose
def check_found_imgs(img_dir):
    try:
        return len(list_img_files(img_dir))
    except:
        return 0

def img_to_b64(img, adjust=False):
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
def process_imgs(img_dir, save_dir, proto, params):
    def iter_cb(imgs, prog):
        img = img_to_b64(img, adjust=True)
        eel.update_img_results(img, prog)
        return eel.is_proc_imgs_stopped()()
    if proto == '0':
        process_fluo_timelapse(img_dir=img_dir, save_dir=save_dir, thr=params['thr'],
            cmax_mult=int(params['cmax_mult']), t_unit=params['t_unit'],
            sbar=params['sbar'], rs=params['rs'], fh_lim=params['fh'],
            cb=params['cb'], thres_mult=params['thres_mult'], iter_cb=iter_cb)
    elif proto == '1':
        process_fluo_images(img_dir=img_dir, save_dir=save_dir, thr=params['thr'],
            cmax_mult=int(params['cmax_mult']), t_unit=params['t_unit'],
            sbar=params['sbar'], rs=params['rs'], fh_lim=params['fh'],
            cb=params['cb'], thres_mult=params['thres_mult'], iter_cb=iter_cb)


if __name__ == '__main__':
    eel.init('gui')
    options = {
        'mode': "chrome",
        'host': 'localhost',
        'port': 8080,
        # 'chromeFlags': ["--start-fullscreen"]
    }

    eel.start('index.html', options=options)
