import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage as ndi
from skimage import img_as_float
from skimage.io import imread
from skimage.measure import regionprops
from skimage.feature import peak_local_max
from skimage.filters import threshold_local, gaussian
from skimage.morphology import disk, dilation, erosion, remove_small_objects
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.segmentation import random_walker, clear_border, find_boundaries

from cytomata.process import track
from cytomata.utils.io import list_fnames, setup_dirs
from cytomata.utils.visual import plot, imshow


def get_median_intensity(img):
    img = img_as_float(img)
    sigma = estimate_sigma(img)
    den = denoise_nl_means(img, h=sigma, sigma=sigma, multichannel=False)
    gau = gaussian(den, sigma=50)
    sub = den - gau
    sub[sub < 0] = 0
    return np.median(sub[sub.nonzero()])


def images_to_median_frame_intensities(img_dir, save_path=None):
    med_ints = []
    for fn in list_fnames(img_dir):
        img = imread(os.path.join(img_dir, fn))
        med_int = get_median_intensity(img)
        med_ints.append(med_int)
    med_ints = np.array(med_ints)
    if save_path is not None:
        np.savetxt(save_path, med_ints, delimiter=',', header='t', comments='')
    return med_ints


def get_regions(img, save_dir=None, gauss_sigma=45,
    autocontrast=False, thres_block=45, thres_offset=0, peaks_min_dist=30, **kwargs):
    img = img_as_float(img)
    sigma = estimate_sigma(img)
    den = denoise_nl_means(img, h=sigma, sigma=sigma, multichannel=False)
    ga0 = gaussian(den, sigma=gauss_sigma)
    sub = den - ga0
    sub[sub < 0] = 0
    if autocontrast:
        sub = equalize_adapthist(sub, clip_limit=0.005)
    th = threshold_local(sub, block_size=thres_block, offset=thres_offset, method='mean')
    thres = gaussian(sub, sigma=1) > th
    labels, _ = ndi.label(thres)
    dist = ndi.distance_transform_edt(sub)
    lmax = peak_local_max(image=dist, labels=labels,
        min_distance=peaks_min_dist, indices=False, exclude_border=False)
    markers, n = ndi.label(dilation(lmax, disk(3)))
    markers[~thres] = -1
    rw = random_walker(sub, markers, beta=100, mode='bf')
    rw[rw < 0] = 0
    min_area = np.quantile([prop.area for prop in regionprops(rw)], 0.25)
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
            plt.imsave(img_path, step_img, cmap='viridis')
    return dict(zip(step_names, step_imgs))


def images_to_ave_single_cell_intensities(img_dir, save_dir=None, **reg_params):
    tracker = track.Sort(max_age=3, min_hits=1)
    trajs = defaultdict(defaultdict(dict).copy)
    for i, fn in enumerate(list_fnames(img_dir)):
        img = imread(os.path.join(img_dir, fn))
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
    img_names = list_fnames(img_dir)
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
            ax.imshow(imread(os.path.join(img_dir, img_names[tframe])))
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
