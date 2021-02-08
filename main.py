import os
import sys
import eel
import warnings
from base64 import b64encode

import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.exposure import equalize_adapthist
import matplotlib.pyplot as plt

from cytomata.utils import list_img_files, setup_dirs
from scripts.process_imgs import process_fluo_timelapse, process_fluo_images


@eel.expose
def guess_img_dir():
    return os.path.join(os.path.expanduser('~'), 'Documents', 'project', 'imgs')

@eel.expose
def guess_save_dir():
    return os.path.join(os.path.expanduser('~'), 'Documents', 'project', 'results')

@eel.expose
def guess_mask_dir():
    return os.path.join(os.path.expanduser('~'), 'Documents', 'project', 'masks')

@eel.expose
def guess_stim_csv():
    return os.path.join(os.path.expanduser('~'), 'Documents', 'project', 'u.csv')

@eel.expose
def check_dir_exists(dir):
    return os.path.exists(dir)

@eel.expose
def check_found_imgs(img_dir):
    try:
        return len(list_img_files(img_dir))
    except:
        return 0

def img_to_b64(img):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return b64encode(cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]).decode('utf-8')

@eel.expose
def process_imgs(img_dir, save_dir, proto, params):
    def iter_cb(img, prog):
        img = img_to_b64(img)
        eel.update_img_results(img, prog)
        return eel.is_proc_imgs_stopped()()
    if params['auto_cmax']:
        params['cmax'] = None
    else:
        params['cmax'] = float(params['cmax'])
    if params['remove_small'] == '':
        params['remove_small'] = None
    else:
        params['remove_small'] = int(params['remove_small'])
    if params['fill_holes'] == '':
        params['fill_holes'] = None
    else:
        params['fill_holes'] = int(params['fill_holes'])
    if params['clear_border'] == '':
        params['clear_border'] = None
    else:
        params['clear_border'] = int(params[clear_border])
    params['segmt_factor'] = float(params['segmt_factor'])
    if proto == '0':
        process_fluo_timelapse(img_dir=img_dir, save_dir=save_dir,
            u_csv=params['stim_csv'],
            t_unit=params['t_unit'], ulabel='Stim.',
            sb_microns=params['sb_microns'], cmax=params['cmax'],
            segmt=params['segmt'], segmt_dots=params['segmt_dots'],
            segmt_mask=params['segmt_mask'], segmt_factor=params['segmt_factor'],
            remove_small=params['remove_small'], fill_holes=params['fill_holes'],
            clear_border=params['clear_border'], adj_bright=params['adj_bright'], iter_cb=iter_cb)
    elif proto == '1':
        process_fluo_images(img_dir=img_dir, save_dir=save_dir,
            sb_microns=params['sb_microns'], cmax=params['cmax'],
            segmt=params['segmt'], segmt_dots=params['segmt_dots'],
            segmt_mask_dir=params['segmt_mask'], segmt_factor=params['segmt_factor'],
            remove_small=params['remove_small'], fill_holes=params['fill_holes'],
            clear_border=params['clear_border'], iter_cb=iter_cb)


if __name__ == '__main__':
    eel.init('gui')
    eel.start('index.html')
