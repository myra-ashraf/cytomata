import os
import sys
import eel
import warnings
from base64 import b64encode

import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.exposure import equalize_adapthist

from cytomata.utils.io import list_img_files, setup_dirs
from cytomata.process.extract import run_frame_ave_analysis, run_single_cell_analysis


@eel.expose
def guess_img_dir():
    return os.path.join(os.path.expanduser('~'), 'Documents', 'imgs')

@eel.expose
def guess_out_dir():
    return os.path.join(os.path.expanduser('~'), 'Documents', 'output')

@eel.expose
def check_dir_exists(dir):
    return os.path.exists(dir)

@eel.expose
def check_found_imgs(img_dir):
    try:
        return len(list_img_files(img_dir))
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
    def iter_cb(imgs, prog):
        irow = int(np.argmax(np.var(imgs[0], axis=1)))
        imgs[0][irow, :] = np.amax(imgs[0])
        img0 = img_to_b64(imgs[0], touchup=True)
        img1 = img_to_b64(imgs[1], touchup=False)
        img2 = img_to_b64(imgs[2], touchup=True)
        img3 = img_to_b64(imgs[3], touchup=False)
        int_ranges = [np.round(np.amin(imgs[0]), 5), np.round(np.amax(imgs[0]), 5),
            np.round(np.amin(imgs[2]), 5), np.round(np.amax(imgs[2]), 5)]
        eel.update_img_results(img0, img1, img2, img3, int_ranges, prog)
        return eel.is_proc_imgs_stopped()()
    if proto == '0':
        run_frame_ave_analysis(img_dir=img_dir, save_dir=out_dir,
            block=int(params['thres_block']), offset=float(params['sub_offset']),
            denoise=params['denoise'], iter_cb=iter_cb, overwrite=True)
    elif proto == '1':
        run_single_cell_analysis(img_dir, save_dir=out_dir, block=int(params['thres_block']),
            offset=float(params['sub_offset']), min_peaks_dist=int(params['det_sens']),
            min_traj_len=int(params['traj_len']), denoise=params['denoise'],
            iter_cb=iter_cb, overwrite=True)


if __name__ == '__main__':
    eel.init('gui')
    options = {
        'mode': "chrome",
        'host': 'localhost',
        'port': 8080,
        'chromeFlags': ["--start-fullscreen"]
    }

    eel.start('index.html', options=options)
