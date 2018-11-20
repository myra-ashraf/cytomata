import os
import time

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.io import imread, imsave

sns.set_style('whitegrid')
plt.rcParams['image.cmap'] = 'viridis'
plt.rcParams['figure.figsize'] = 12, 8
plt.rcParams['text.color'] = '#212121'
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['axes.titlepad'] = 18
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.labelpad'] = 10
plt.rcParams['axes.labelcolor'] = '#212121'
plt.rcParams['axes.labelweight'] = 600
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.edgecolor'] = '#212121'
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['lines.linewidth'] = 3
sns.set_palette(['#1E88E5', '#43A047', '#e53935', '#5E35B1', '#FFB300', '#00ACC1', '#3949AB', '#F4511E'])


def plot(x, y, labels, xlabel, ylabel, title, color=None, save_path=None):
    plt.plot(x, y, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    labels = [labels] if type(labels) is str else labels
    if labels:
        plt.legend(labels=labels, loc='best')
    plt.title(title, loc='left')
    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def imshow(img, title, save_path=None):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(title)
    ax.grid(False)
    ax.axis('off')
    if save_path is not None:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)


def convert_to_png(img_dir, png_dir):
    for i, img in enumerate(list_img_names(img_dir)):
        img = imread(os.path.join(img_dir, img), as_gray=True)
        img = exposure.equalize_adapthist(img, clip_limit=0.003)
        imsave(os.path.join(png_dir, str(i) + '.png'), img)


def frames_to_video(img_dir, vid_path, fps):
    images = list_img_names(img_dir)
    frame0 = cv2.imread(os.path.join(img_dir, images[0]))
    height, width = frame0.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_path = vid_path + '.mp4' if not vid_path.endswith('.mp4') else vid_path
    video = cv2.VideoWriter(vid_path, fourcc, fps, (width, height))
    for image in images:
        image = cv2.imread(os.path.join(img_dir, image))
        video.write(image)
    cv2.destroyAllWindows()
    video.release()


class DynamicPlot(object):
    """"""
    def __init__(self, x, y, labels, xlabel, ylabel, title, save_dir=None):
        self.save_dir = save_dir
        self.fig, self.ax = plt.subplots()
        self.lines = self.ax.plot(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        labels = [labels] if type(labels) is str else labels
        plt.legend(labels=labels, loc='best')
        plt.title(title, loc='left')
        self.fig.canvas.draw()
        self.count = 0
        plt.show(block=False)
        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
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
