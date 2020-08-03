import os
import imghdr

import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from scipy.interpolate import interp1d
from skimage import img_as_ubyte
from cytomata import turbo_cmap


custom_palette = [
    '#1976D2', '#D32F2F', '#388E3C',
    '#7B1FA2', '#F57C00', '#C2185B',
    '#FBC02D', '#303F9F', '#0097A7',
    '#5D4037', '#455A64', '#AFB42B']


custom_styles = {
    'image.cmap': 'viridis',
    'figure.figsize': (16, 8),
    'text.color': '#212121',
    'axes.titleweight': 'bold',
    'axes.titlesize': 32,
    'axes.titlepad': 18,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 28,
    'axes.labelpad': 10,
    'axes.labelcolor': '#212121',
    'axes.labelweight': 600,
    'axes.linewidth': 3,
    'axes.edgecolor': '#212121',
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'legend.fontsize': 28,
    'lines.linewidth': 5
}


def list_img_files(dir):
    return [os.path.join(dir, fn) for fn in natsorted(os.listdir(dir), key=lambda y: y.lower())
        if imghdr.what(os.path.join(dir, fn)) in ['tiff', 'jpeg', 'png', 'gif']]


def setup_dirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def clear_screen():
    os.system('cls' if os.name=='nt' else 'clear')


def rescale(aa):
    return (aa - min(aa)) / (max(aa) - min(aa))


def sse(aa, bb):
    return np.sum((aa - bb)**2)


def approx_half_life(t, y, phase='fall'):
    """Approximate half life of reaction process using cubic spline interpolation."""
    t = np.array(t)
    y = np.array(y)
    if phase == 'rise':
        tp = t[:y.argmax()]
        yp = y[:y.argmax()]
    elif phase == 'fall':
        tp = t[y.argmax():]
        yp = y[y.argmax():]
    y_half = (np.max(y) - np.min(y))/2
    yf = interp1d(tp, yp, 'cubic')
    ti = np.arange(tp[0], tp[-1], 1)
    yi = yf(ti)
    idx = np.argmin((yi - y_half)**2)
    t_half = ti[idx]
    return t_half


def imgs_to_mp4(imgs, vid_path, fps=10):
    for i, img in enumerate(imgs):
        img = img_as_ubyte(img)
        if i == 0:
            height, width = imgs[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vid_path = vid_path + '.mp4' if not vid_path.endswith('.mp4') else vid_path
            video = cv2.VideoWriter(vid_path, fourcc, fps, (width, height))
        video.write(img)
    cv2.destroyAllWindows()
    video.release()
