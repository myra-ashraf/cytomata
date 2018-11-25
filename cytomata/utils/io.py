import os


def list_fnames(dir, key=float):
    return [fn for fn in sorted(os.listdir(dir), key=key)]


def setup_dirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
