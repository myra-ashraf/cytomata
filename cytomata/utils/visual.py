import os

import cv2
import imageio
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import img_as_ubyte, img_as_float

from cytomata.utils.io import setup_dirs


custom_palette = [
    '#1E88E5', '#43A047', '#e53935',
    '#5E35B1', '#FFB300', '#00ACC1',
    '#3949AB', '#F4511E', '#D81B60']
custom_styles = {
    'image.cmap': 'viridis',
    'figure.figsize': (12, 8),
    'text.color': '#212121',
    'axes.titleweight': 'bold',
    'axes.titlesize': 28,
    'axes.titlepad': 18,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 22,
    'axes.labelpad': 10,
    'axes.labelcolor': '#212121',
    'axes.labelweight': 600,
    'axes.linewidth': 2,
    'axes.edgecolor': '#212121',
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'lines.linewidth': 3
}


def plot(x, y=None, xlabel=None, ylabel=None, title=None, labels=None, show=False,
    figsize=custom_styles['figure.figsize'], legend_loc='best', save_path=None):
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, ax = plt.subplots(figsize=figsize)
        if y is not None:
            ax.plot(x, y)
        else:
            ax.plot(x)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if labels is not None:
            labels = [labels] if type(labels) is str else labels
            ax.legend(labels=labels, loc=legend_loc)
        if title is not None:
            ax.set_title(title, loc='left')
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer._renderer)
        if save_path is not None:
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)
        return img


def imshow(img, title=None, axes=False, colorbar=True, show=False, save_path=None):
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, ax = plt.subplots()
        axim = ax.imshow(img)
        if title is not None:
            ax.set_title(title)
        ax.grid(False)
        if not axes:
            ax.axis('off')
        if colorbar:
            cb = fig.colorbar(axim, pad=0.01)
            cb.outline.set_linewidth(0)
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer._renderer)
        if save_path is not None:
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)
        return img


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


def imgs_to_gif(imgs, gif_path, fps=10):
    with imageio.get_writer(gif_path, mode='I', fps=fps) as writer:
        for img in imgs:
            writer.append_data(img_as_float(img))


class DynamicPlot(object):
    """"""
    def __init__(self, x, y, xlabel, ylabel, title=None, labels=None, save_dir=None):
        self.save_dir = save_dir
        self.fig, self.ax = plt.subplots()
        self.lines = self.ax.plot(x, y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if labels is not None:
            labels = [labels] if type(labels) is str else labels
            ax.legend(labels=labels, loc='best')
        if title is not None:
            ax.set_title(title, loc='left')
        self.fig.canvas.draw()
        self.count = 0
        plt.show(block=False)
        if self.save_dir is not None:
            setup_dirs(self.save_dir)
            save_path = os.path.join(self.save_dir, str(self.count) + '.png')
            self.fig.savefig(save_path, dpi=100)

    def update(self, x, y):
        self.count += 1
        if len(x.shape) > 1:
            for xi, l in zip(x.T , self.lines):
                l.set_xdata(xi)
        else:
            self.lines[0].set_xdata(x)
        if len(y.shape) > 1:
            for yi, l in zip(y.T, self.lines):
                l.set_ydata(yi)
        else:
            self.lines[0].set_ydata(y)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if self.save_dir is not None:
            save_path = os.path.join(self.save_dir, str(self.count) + '.png')
            self.fig.savefig(save_path, dpi=100)

    def close(self):
        plt.close(self.fig)
