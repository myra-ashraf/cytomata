import os
import imghdr


def list_fnames(dir):
    return [fn for fn in sorted(
        os.listdir(dir), key=lambda f: float(''.join(filter(str.isnumeric, f))))
        if imghdr.what(os.path.join(dir, fn)) in ['tiff', 'jpeg', 'png']]


def setup_dirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
