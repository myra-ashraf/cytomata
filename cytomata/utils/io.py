import os


def list_fnames_sorted(dir):
    return [fn for fn in sorted(os.listdir(dir), key=float)]


def setup_dirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
