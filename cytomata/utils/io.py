import os
import imghdr


def list_img_files(dir):
    return [fn for fn in sorted(os.listdir(dir))
        if imghdr.what(os.path.join(dir, fn)) in ['tiff', 'jpeg', 'png', 'gif']]


def setup_dirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
