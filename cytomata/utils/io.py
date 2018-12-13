from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (ascii, bytes, chr, dict, filter, hex, input,
    int, map, next, oct, open, pow, range, round, str, super, zip)

import os
import shutil
import imghdr
import warnings

from natsort import natsorted


def list_img_files(dir):
    return [fn for fn in natsorted(os.listdir(dir), key=lambda y: y.lower())
        if imghdr.what(os.path.join(dir, fn)) in ['tiff', 'jpeg', 'png', 'gif']]


def setup_dirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

def pims_open(dir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ext in ['.tiff', '.tif', '.png', '.jpg', '.jpeg', '.gif', '']:
            try:
                imgs = pims.open(os.path.join(img_dir, '*' + ext))
                break
            except:
                continue
    return imgs
