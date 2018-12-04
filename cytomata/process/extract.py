import os
import json
from collections import defaultdict

import pims
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage as ndi
from skimage import img_as_float
from skimage.io import imsave
from skimage.exposure import equalize_adapthist
from skimage.measure import regionprops
from skimage.feature import peak_local_max
from skimage.filters import threshold_local, gaussian
from skimage.morphology import disk, dilation, erosion, remove_small_objects
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.segmentation import random_walker, clear_border, find_boundaries

from cytomata.process import track
from cytomata.utils.io import setup_dirs
from cytomata.utils.visual import plot, imshow


def get_median_intensity(img, gauss_sigma=50):
    img = img_as_float(img)
    sigma = estimate_sigma(img)
    den = denoise_nl_means(img, h=sigma, sigma=sigma, multichannel=False)
    gau = gaussian(den, sigma=gauss_sigma)
    sub = den - gau
    sub[sub < 0] = 0
    med_int = np.median(sub[sub.nonzero()])
    imgs = [img, den, gau, sub]
    return med_int, imgs


def images_to_median_frame_intensities(img_dir, save_dir=None, gauss_sigma=40, iter_cb=None):
    med_ints = []
    img_step = {0: 'original', 1: 'denoised', 2: 'gaussian', 3: 'subtracted'}
    imgs = pims.open(os.path.join(img_dir, '*'))
    for i, img in enumerate(imgs):
        img = img_as_float(img)
        med_int, imgs = get_median_intensity(img, gauss_sigma)
        med_ints.append(med_int)
        if save_dir is not None:
            for j, img in enumerate(imgs):
                img_step_dir = os.path.join(save_dir, img_step[j])
                setup_dirs(img_step_dir)
                img_path = os.path.join(img_step_dir, str(i) + '.tif')
                # plt.imsave(img_path, img, cmap='viridis')
                imsave(img_path, img)
        if iter_cb is not None:
            fig, ax = plt.subplots(figsize=(16, 8))
            ax.plot(range(len(med_ints)), med_ints)
            ax.set_xlabel('Frame')
            ax.set_ylabel('Median Intensity')
            fig.tight_layout(pad=0)
            fig.canvas.draw()
            iter_imgs = imgs + [np.array(fig.canvas.renderer._renderer)]
            plt.close(fig)
            prog = int(round((i+1)/len(img_fnames) * 100))
            if iter_cb(med_int, iter_imgs, prog):
                break
    if save_dir is not None:
        plot_path = os.path.join(save_dir, 'med_ints.png')
        plot(range(len(med_ints)), med_ints, 'Frames', 'Median Intensity', save_path=plot_path)
        csv_path = os.path.join(save_dir, 'med_ints.csv')
        np.savetxt(csv_path, np.array(med_ints), delimiter=',', header='median_intensity', comments='')
    return med_ints


def get_regions(img, save_dir=None, gauss_sigma=30,
    autocontrast=False, thres_block=25, thres_offset=-0.005, peaks_min_dist=25, **kwargs):
    img = img_as_float(img)
    sigma = estimate_sigma(img)
    den = denoise_nl_means(img, h=sigma, sigma=sigma, multichannel=False)
    ga0 = gaussian(den, sigma=gauss_sigma)
    sub = den - ga0
    sub[sub < 0] = 0
    cont = equalize_adapthist(sub, clip_limit=0.001)
    th = threshold_local(cont, block_size=thres_block, offset=thres_offset, method='mean')
    thres = dilation(erosion(gaussian(cont, sigma=1) > th, disk(5)), disk(4))
    labels, _ = ndi.label(thres)
    dist = ndi.distance_transform_edt(cont)
    lmax = peak_local_max(image=dist, labels=labels,
        min_distance=peaks_min_dist, indices=False, exclude_border=False)
    markers, n = ndi.label(dilation(lmax, disk(3)))
    markers[~thres] = -1
    rw = random_walker(cont, markers, beta=1000, mode='bf')
    rw[rw < 0] = 0
    min_area = np.quantile([prop.area for prop in regionprops(rw)], 0.10)
    regions = clear_border(remove_small_objects(rw, min_size=min_area), buffer_size=0)
    bouns = find_boundaries(regions)
    overlay = sub.copy()
    overlay[bouns] = np.percentile(overlay, 99.99)
    step_names = ['original', 'denoised', 'subtracted', 'thresholded', 'peaks', 'regions', 'overlay']
    step_imgs = [img, den, sub, thres, markers, regions, overlay]
    if save_dir is not None:
        for i, (step_name, step_img) in enumerate(zip(step_names, step_imgs)):
            res_dir = os.path.join(save_dir, str(i) + '_' + step_name)
            setup_dirs(res_dir)
            img_path = os.path.join(res_dir, str(len(os.listdir(res_dir))) + '.png')
            # plt.imsave(img_path, step_img, cmap='viridis')
            imsave(img_path, step_img)
    return dict(zip(step_names, step_imgs))


def images_to_ave_single_cell_intensities(img_dir, save_dir=None, **reg_params):
    tracker = track.Sort(max_age=3, min_hits=1)
    trajs = defaultdict(defaultdict(dict).copy)
    imgs = pims.open(os.path.join(img_dir, '*'))
    for i, img in enumerate(imgs):
        img = img_as_float(img)
        regions_dir = os.path.join(save_dir, 'regions')
        setup_dirs(regions_dir)
        res = get_regions(img, regions_dir, **reg_params)
        ints = np.array([prop.mean_intensity
            for prop in regionprops(res['regions'], res['subtracted'])])
        dets = np.array([prop.bbox for prop in regionprops(res['regions'])])
        trks = tracker.update(dets)
        for trk in trks:  # Restructure data trajectory-wise
            id = int(trk[4])
            ind = np.argmax(np.array([track.iou(det, trk) for det in dets]))
            if len(trajs[id]['color']) == 3:
                ecolor = trajs[id]['color']
            else:
                ecolor = np.random.rand(3,)
            trajs[id]['color'] = list(ecolor)
            trajs[id]['dets'][i] = list(dets[ind])
            trajs[id]['trks'][i] = list(trk[0:4])
            trajs[id]['ints'][i] = ints[ind]
    with open(os.path.join(save_dir, 'trajs.json'), 'w') as fp:
        json.dump(trajs, fp)
    return trajs


def plot_single_cell_trajectories(trajs, save_dir, min_traj_length=100):
    trajs = {id:info for id, info in trajs.items() if len(info['ints']) > min_traj_length}
    for frames_ints in [info['ints'] for info in trajs.values()]:
        fnums, fints = zip(*frames_ints.items())
        plt.plot(fnums, fints)
    plt.ylabel('Ave Region Intensity')
    plt.xlabel('Frame')
    plt.title('Trajectories')
    fig_name = 'min_' + str(min_traj_length) + '_trajs.png'
    plt.savefig(os.path.join(save_dir, fig_name), bbox_inches='tight')
    plt.close()


def plot_single_cell_trajectories_ave(trajs, save_dir, by_frame=True):
    trajs_frames_ints = pd.DataFrame([info['ints'] for info in trajs.values()])
    traj_ave = trajs_frames_ints.mean(axis=0)
    traj_std = trajs_frames_ints.std(axis=0)
    plt.fill_between(
        range(len(traj_ave)), traj_ave + traj_std,
        traj_ave - traj_std, alpha=0.5, color='#1E88E5')
    plt.plot(traj_ave, color='#1E88E5')
    plt.ylabel('Ave Intensity')
    plt.xlabel('Frame')
    plt.title('Frame Ave Intensity of Trajectories')
    plt.savefig(os.path.join(save_dir, 'trajs_frame_aves.png'), bbox_inches='tight')
    plt.close()


def save_single_cell_data(img_dir, trajs, save_dir, min_traj_length=100, calc_func=None, **kwargs):
    trajs = {id:info for id, info in trajs.items() if len(info['ints']) > min_traj_length}
    df = pd.DataFrame(columns=trajs.keys())
    for id, info in trajs.items():
        for frame, fint in info['ints'].items():
            df.loc[frame, id] = fint
    df.sort_index(inplace=True)
    df.to_csv(os.path.join(save_dir, 'frames_ints.csv'), index=False)
    imgs = pims.open(os.path.join(img_dir, '*'))
    for id, info in trajs.items():
        traj_dir = os.path.join(save_dir, 'traj')
        id_dir = os.path.join(traj_dir, str(id))
        setup_dirs(id_dir)
        frames = []
        ints = []
        for i, (tframe, tbbox) in enumerate(info['trks'].items()):
            fig, ax = plt.subplots()
            flint = info['ints'][tframe]
            frames.append(tframe)
            ints.append(flint)
            ax.imshow(imgs[tframe])
            ax.axis('off')
            rect = Rectangle((tbbox[1], tbbox[0]), tbbox[3] - tbbox[1],
                tbbox[2] - tbbox[0], fill=False, edgecolor=info['color'], linewidth=1)
            ax.add_patch(rect)
            ax.set_title('Frame: ' + str(tframe) + ' | ' + 'Intensity: ' + str(round(flint, 4)))
            if i == 0:
                plt.savefig(os.path.join(traj_dir, str(id) + '.png'), bbox_inches='tight')
            plt.savefig(os.path.join(id_dir, str(tframe) + '.png'), bbox_inches='tight')
            plt.close(fig)
        if calc_func is not None:
            calc = calc_func(frames, ints, **kwargs)
            calc_path = os.path.join(id_dir, 'calcs.csv')
            np.savetxt(calc_path, np.array(calc), delimiter=',', header='calc', comments='')
        traj_path = os.path.join(id_dir, 'traj_int.png')
        plot(frames, ints, labels=None, xlabel='Frame',
            ylabel='Ave Intensity', title='Trajectory #' + str(id), save_path=traj_path)
        flint_path = os.path.join(id_dir, 'flint.csv')
        np.savetxt(flint_path, np.column_stack((frames, ints)), delimiter=',', header='frame,fl_int', comments='')
    plt.close('all')
