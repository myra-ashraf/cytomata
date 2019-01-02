import os
import json
import shutil
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage as ndi
from skimage import img_as_float, img_as_uint
from skimage.io import imread, imsave
from skimage.exposure import equalize_adapthist
from skimage.measure import regionprops
from skimage.feature import peak_local_max
from skimage.filters import threshold_local, gaussian
from skimage.morphology import disk, dilation, erosion, remove_small_objects
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.segmentation import random_walker, clear_border, find_boundaries

from cytomata.process import track
from cytomata.utils.io import list_img_files, setup_dirs
from cytomata.utils.visual import plot, custom_styles, custom_palette


def bkg_subtract(img, block=201, offset=1.0, denoise=False, prof_row=None):
    img = img_as_float(img)
    sigma = estimate_sigma(img)
    bkg = threshold_local(img, block_size=block, method='gaussian', offset=-sigma*offset)
    sub = img - bkg
    if denoise:
        sub = ndi.median_filter(sub, size=5)
    sub[sub < 0.0] = 0.0
    sub = sub - np.amin(sub)
    ave_int = 0.0
    no_edge = sub[10:-10, 10:-10]
    if np.count_nonzero(no_edge) > 0:
        ave_int = np.mean(no_edge[no_edge.nonzero()])
    if prof_row is None:
        prof_row = int(np.argmax(np.var(img, axis=1)))
    bkg_prof = plot(range(len(img)), np.column_stack((img[prof_row, :], bkg[prof_row, :])),
        xlabel='Image Column', ylabel='Pixel Intensity',
        title='Intensity Profile Img[' + str(prof_row) + ', :]',
        labels=['Original', 'Background'], legend_loc='upper left', figsize=(11, 6))
    sub_prof = plot(range(len(img)), sub[prof_row, :],
        xlabel='Image Column', ylabel='Pixel Intensity',
        title='Intensity Profile Img[' + str(prof_row) + ', :]',
        labels=['Subtracted'], legend_loc='upper left', figsize=(11, 6))
    results = {'ori_img': img, 'bkg_img': bkg, 'sub_img': sub,
        'ave_int': ave_int, 'bkg_prof': bkg_prof, 'sub_prof': sub_prof}
    return results


def run_frame_ave_analysis(img_dir, save_dir=None, block=201,
    offset=1.0, denoise=False, stylize=False, iter_cb=None, overwrite=True):
    if save_dir is not None:
        if overwrite and os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        ori_img_dir = os.path.join(save_dir, '0-original')
        setup_dirs(ori_img_dir)
        bkg_img_dir = os.path.join(save_dir, '1-background')
        setup_dirs(bkg_img_dir)
        sub_img_dir = os.path.join(save_dir, '2-subtracted')
        setup_dirs(sub_img_dir)
        bkg_prof_dir = os.path.join(save_dir, 'bkg_profile')
        setup_dirs(bkg_prof_dir)
        sub_prof_dir = os.path.join(save_dir, 'sub_profile')
        setup_dirs(sub_prof_dir)
    ave_ints = []
    imgfs = list_img_files(img_dir)
    for i, imgf in enumerate(imgfs):
        results = bkg_subtract(imread(imgf), block=block, offset=offset, denoise=denoise)
        ave_ints.append(results['ave_int'])
        if save_dir is not None:
            imsave(os.path.join(bkg_prof_dir, str(i) + '.png'), results['bkg_prof'])
            imsave(os.path.join(sub_prof_dir, str(i) + '.png'), results['sub_prof'])
            if stylize:
                plt.imsave(os.path.join(ori_img_dir, str(i) + '.png'),
                    results['ori_img'], cmap='viridis')
                plt.imsave(os.path.join(bkg_img_dir, str(i) + '.png'),
                    results['bkg_img'], cmap='viridis')
                plt.imsave(os.path.join(sub_img_dir, str(i) + '.png'),
                    results['sub_img'], cmap='viridis')
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    imsave(os.path.join(ori_img_dir, str(i) + '.tiff'),
                        img_as_uint(results['ori_img']))
                    imsave(os.path.join(bkg_img_dir, str(i) + '.tiff'),
                        img_as_uint(results['bkg_img']))
                    imsave(os.path.join(sub_img_dir, str(i) + '.tiff'),
                        img_as_uint(results['sub_img']))
        if iter_cb is not None:
            plt_int = plot(range(len(ave_ints)), ave_ints,
                xlabel='Frame', title='Ave Frame Intensity (Nonzero Pixels)', figsize=(11, 6))
            iter_imgs = [results['ori_img'], results['bkg_prof'], results['sub_img'], plt_int]
            progress = (i + 1)/len(imgfs) * 100
            if iter_cb(iter_imgs, progress):
                break
    if save_dir is not None:
        plot(range(len(ave_ints)), ave_ints,
            xlabel='Frame', ylabel='Ave Intensity (Nonzero Pixels)',
            save_path= os.path.join(save_dir, 'ave_ints.png'))
        if ave_ints[0] > 0:
            plot(range(len(ave_ints)), np.array(ave_ints)/ave_ints[0],
                xlabel='Frame', ylabel='Fold Change',
                save_path= os.path.join(save_dir, 'fold_change.png'))
        csv_path = os.path.join(save_dir, 'ave_ints.csv')
        np.savetxt(csv_path, np.array(ave_ints), delimiter=',', header='ave_intensity', comments='')
    return ave_ints


def detect_regions(img, block=201, offset=1.0, denoise=False, min_peaks_dist=25):
    pre = bkg_subtract(img, block=block, offset=offset, denoise=denoise)
    # cont = equalize_adapthist(sub, clip_limit=0.001)
    # th = threshold_local(cont, block_size=block, offset=offset, method='mean')
    # sigma = estimate_sigma(pre['sub_img'])
    # den = denoise_nl_means(pre['sub_img'], h=sigma, sigma=sigma, multichannel=False)
    if denoise:
        den = pre['sub_img']
    else:
        den = ndi.median_filter(pre['sub_img'], size=9)
    thr = erosion(den > 0, disk(2))
    labels, _ = ndi.label(thr)
    dist = ndi.distance_transform_edt(den)
    plm = peak_local_max(image=dist, labels=labels, min_distance=min_peaks_dist, indices=False)
    markers, n = ndi.label(dilation(plm, disk(3)))
    markers[~thr] = -1
    rw = random_walker(den, markers, beta=1000, mode='bf')
    rw[rw < 0] = 0
    min_area = np.quantile([prop.area for prop in regionprops(rw)], 0.10)
    reg = clear_border(remove_small_objects(rw, min_size=min_area), buffer_size=0)
    bnd = den.copy()
    bnd[find_boundaries(reg)] = np.percentile(bnd, 99.99)
    results = {'ori_img': pre['ori_img'], 'bkg_img': pre['bkg_img'],
        'sub_img': pre['sub_img'], 'bkg_prof': pre['bkg_prof'], 'sub_prof': pre['sub_prof'],
        'den_img': den, 'thr_img': thr, 'pks_img': dist, 'reg_img': reg, 'bnd_img': bnd}
    return results


def filter_trajectories(trajs, min_traj_len=100):
    return {id:info for id, info in trajs.items() if len(info['trks']) > min_traj_len}


def plot_single_cell_trajectories(trajs, save_path=None, figsize=(12, 8)):
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, ax = plt.subplots(figsize=figsize)
        for frames_ints in (info['ints'] for info in trajs.values()):
            fnums, fints = zip(*frames_ints.items())
            plt.plot(fnums, fints)
        plt.xlabel('Frame')
        plt.title('Trajectories', loc='left')
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer._renderer)
        plt.ylabel('Ave Region Intensity')
        if save_path is not None:
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return img


def plot_ave_cell_trajectory(trajs, save_path=None, by_frame=False, figsize=(12, 8)):
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, ax = plt.subplots(figsize=figsize)
        trajs = pd.DataFrame([info['ints'] for info in trajs.values()])
        if not by_frame:
            trajs = trajs.apply(lambda x: pd.Series(x.dropna().values))
        traj_ave = trajs.mean(axis=0)
        traj_std = trajs.std(axis=0)
        plt.fill_between(
            range(len(traj_ave)), traj_ave + traj_std,
            traj_ave - traj_std, alpha=0.5, color='#1E88E5')
        plt.plot(traj_ave, color='#1E88E5')
        plt.ylabel('Ave Intensity')
        plt.xlabel('Frame')
        plt.title('Ave Intensity of All Trajectories', loc='left')
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer._renderer)
        if save_path is not None:
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return img


def save_all_traj_data(trajs, save_dir):
    with open(os.path.join(save_dir, 'trajs.json'), 'w') as fp:
        json.dump(trajs, fp)
    df = pd.DataFrame(columns=trajs.keys())
    for id, info in trajs.items():
        for frame, fint in info['ints'].items():
            df.loc[frame, id] = fint
    df.sort_index(inplace=True)
    df.to_csv(os.path.join(save_dir, 'frames_ints.csv'), index=False)


def save_each_traj_data(trajs, img_dir, save_dir):
    imgfs = list_img_files(img_dir)
    for id, info in trajs.items():
        traj_dir = os.path.join(save_dir, 'traj')
        id_dir = os.path.join(traj_dir, str(id))
        setup_dirs(id_dir)
        frames = []
        ints = []
        # Trajectory images with bounding boxes
        for i, (tframe, tbbox) in enumerate(info['trks'].items()):
            fig, ax = plt.subplots()
            flint = info['ints'][tframe]
            frames.append(tframe)
            ints.append(flint)
            ax.imshow(imread(imgfs[tframe]))
            ax.axis('off')
            rect = Rectangle((tbbox[1], tbbox[0]), tbbox[3] - tbbox[1],
                tbbox[2] - tbbox[0], fill=False, edgecolor=info['color'], linewidth=1)
            ax.add_patch(rect)
            ax.set_title('Frame: ' + str(tframe) + ' | ' + 'Intensity: ' + str(round(flint, 4)))
            if i == 0:
                plt.savefig(os.path.join(traj_dir, str(id) + '.png'), bbox_inches='tight')
            plt.savefig(os.path.join(id_dir, str(tframe) + '.png'), bbox_inches='tight')
            plt.close(fig)
        # Single trajectory graph plus raw data
        traj_path = os.path.join(id_dir, 'traj_int.png')
        plot(frames, ints, 'Frame', 'Ave Intensity',
            title='Trajectory #' + str(id), save_path=traj_path)
        flint_path = os.path.join(id_dir, 'flint.csv')
        np.savetxt(flint_path, np.column_stack((frames, ints)),
            delimiter=',', header='frame,fl_int', comments='')
    plt.close('all')


def run_single_cell_analysis(img_dir, save_dir=None, block=201, offset=1.0,
    min_peaks_dist=25, min_traj_len=10, denoise=False, stylize=False, iter_cb=None, overwrite=False):
    if save_dir is not None:
        if overwrite and os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        ori_img_dir = os.path.join(save_dir, '0-original')
        setup_dirs(ori_img_dir)
        bkg_img_dir = os.path.join(save_dir, '1-background')
        setup_dirs(bkg_img_dir)
        sub_img_dir = os.path.join(save_dir, '2-subtracted')
        setup_dirs(sub_img_dir)
        den_img_dir = os.path.join(save_dir, '3-denoised')
        setup_dirs(den_img_dir)
        thr_img_dir = os.path.join(save_dir, '4-threshold')
        setup_dirs(thr_img_dir)
        pks_img_dir = os.path.join(save_dir, '5-peaks')
        setup_dirs(pks_img_dir)
        reg_img_dir = os.path.join(save_dir, '6-regions')
        setup_dirs(reg_img_dir)
        bnd_img_dir = os.path.join(save_dir, '7-boundaries')
        setup_dirs(bnd_img_dir)
        bkg_prof_dir = os.path.join(save_dir, 'bkg_profile')
        setup_dirs(bkg_prof_dir)
        sub_prof_dir = os.path.join(save_dir, 'sub_profile')
        setup_dirs(sub_prof_dir)
    tracker = track.Sort(max_age=3, min_hits=1)
    trajs = defaultdict(defaultdict(dict).copy)
    imgfs = list_img_files(img_dir)
    for i, imgf in enumerate(imgfs):
        img = imread(imgf)
        results = detect_regions(img,
            block=block, offset=offset, denoise=denoise, min_peaks_dist=min_peaks_dist)
        if save_dir is not None:
            plt.imsave(os.path.join(bkg_prof_dir, str(i) + '.png'),
                results['bkg_prof'], cmap='viridis')
            plt.imsave(os.path.join(sub_prof_dir, str(i) + '.png'),
                results['sub_prof'], cmap='viridis')
            plt.imsave(os.path.join(thr_img_dir, str(i) + '.png'),
                results['thr_img'], cmap='viridis')
            plt.imsave(os.path.join(pks_img_dir, str(i) + '.png'),
                results['pks_img'], cmap='viridis')
            plt.imsave(os.path.join(reg_img_dir, str(i) + '.png'),
                results['reg_img'], cmap='viridis')
            plt.imsave(os.path.join(bnd_img_dir, str(i) + '.png'),
                results['bnd_img'], cmap='viridis')
            if stylize:
                plt.imsave(os.path.join(ori_img_dir, str(i) + '.png'),
                    results['ori_img'], cmap='viridis')
                plt.imsave(os.path.join(bkg_img_dir, str(i) + '.png'),
                    results['bkg_img'], cmap='viridis')
                plt.imsave(os.path.join(sub_img_dir, str(i) + '.png'),
                    results['sub_img'], cmap='viridis')
                plt.imsave(os.path.join(den_img_dir, str(i) + '.png'),
                    results['den_img'], cmap='viridis')
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    imsave(os.path.join(ori_img_dir, str(i) + '.tiff'),
                        img_as_uint(results['ori_img']))
                    imsave(os.path.join(bkg_img_dir, str(i) + '.tiff'),
                        img_as_uint(results['bkg_img']))
                    imsave(os.path.join(sub_img_dir, str(i) + '.tiff'),
                        img_as_uint(results['sub_img']))
                    imsave(os.path.join(den_img_dir, str(i) + '.tiff'),
                        img_as_uint(results['den_img']))
        ints = np.array([prop.mean_intensity
            for prop in regionprops(results['reg_img'], results['sub_img'])])
        dets = np.array([prop.bbox for prop in regionprops(results['reg_img'])])
        trks = tracker.update(dets)
        for trk in trks:  # Restructure data trajectory-wise
            id = int(trk[4])
            ind = np.argmax(np.array([track.iou(det, trk) for det in dets]))
            if len(trajs[id]['color']) == 3:
                ecolor = trajs[id]['color']
            else:
                ecolor = list(np.random.rand(3,).astype(float))
            trajs[id]['color'] = ecolor
            trajs[id]['dets'][i] = [int(bb) for bb in dets[ind]]
            trajs[id]['trks'][i] = [float(tt) for tt in trk[0:4]]
            trajs[id]['ints'][i] = float(ints[ind])
        if iter_cb is not None:
            plt_trajs = plot_single_cell_trajectories(trajs, figsize=(11, 6))
            iter_imgs = [results['ori_img'], results['bkg_prof'], results['bnd_img'], plt_trajs]
            progress = (i + 1)/len(imgfs) * 100
            if iter_cb(iter_imgs, progress):
                break
    if save_dir is not None:
        if min_traj_len > len(imgfs):
            min_traj_len = len(imgfs)
        trajs_filtered = filter_trajectories(trajs, min_traj_len)
        plot_single_cell_trajectories(trajs_filtered, save_path=os.path.join(save_dir, 'trajs.png'))
        plot_ave_cell_trajectory(trajs_filtered, save_path=os.path.join(save_dir, 'ave_trajs.png'))
        save_all_traj_data(trajs_filtered, save_dir)
        save_each_traj_data(trajs_filtered, img_dir, save_dir)
    return trajs_filtered
