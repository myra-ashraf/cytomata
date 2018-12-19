import os
import json
import shutil
import warnings
from collections import defaultdict

import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage as ndi
from skimage import img_as_float, img_as_uint
from skimage.io import imread, imsave
from skimage.exposure import equalize_adapthist
from skimage.measure import regionprops
from skimage.feature import peak_local_max
from skimage.filters import threshold_local, gaussian
from skimage.morphology import disk, dilation, erosion, remove_small_objects, white_tophat
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.segmentation import random_walker, clear_border, find_boundaries

from cytomata.process import track
from cytomata.utils.io import list_img_files, setup_dirs
from cytomata.utils.visual import plot


def get_ave_intensity(img, block=201, offset=0, denoise=True, debug_row=None):
    img = img_as_float(img)
    sigma = estimate_sigma(img)
    if debug_row is None:
        debug_row = int(np.argmax(np.var(img, axis=1)))
    if denoise:
        den = denoise_nl_means(img, h=sigma, sigma=sigma, multichannel=False)
        sigma = estimate_sigma(den)
        bkg = threshold_local(den, block_size=block, method='gaussian', offset=-sigma*offset)
        sub = den - bkg
        bkg_prof = plot(range(len(img)), np.column_stack((den[debug_row, :], bkg[debug_row, :])),
            xlabel='Image Column', ylabel='Pixel Intensity',
            title='Intensity Profile Img[' + str(debug_row) + ', :]',
            labels=['Denoised', 'Background'], legend_loc='upper left', figsize=(11, 6))
    else:
        den = None
        bkg = threshold_local(img, block_size=block, method='gaussian', offset=-sigma*offset)
        sub = img - bkg
        bkg_prof = plot(range(len(img)), np.column_stack((img[debug_row, :], bkg[debug_row, :])),
            xlabel='Image Column', ylabel='Pixel Intensity',
            title='Intensity Profile Img[' + str(debug_row) + ', :]',
            labels=['Original', 'Background'], legend_loc='upper left', figsize=(11, 6))
    sub[sub < 0.0] = 0.0
    sub = sub - np.amin(sub)
    ave_int = 0.0
    if np.count_nonzero(sub) > 0:
        ave_int = np.mean(sub[sub.nonzero()])
    sub_prof = plot(range(len(img)), sub[debug_row, :],
        xlabel='Image Column', ylabel='Pixel Intensity',
        title='Intensity Profile Img[' + str(debug_row) + ', :]',
        labels=['Subtracted'], legend_loc='upper left', figsize=(11, 6))
    results = {'ave_int': ave_int, 'ori_img': img, 'den_img': den, 'bkg_img': bkg,
        'sub_img': sub, 'bkg_prof': bkg_prof, 'sub_prof': sub_prof}
    return results


def images_to_ave_frame_intensities(img_dir, save_dir=None, block=201,
    offset=0, denoise=True, stylize=False, iter_cb=None, overwrite=True):
    if save_dir is not None:
        if overwrite and os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        ori_img_dir = os.path.join(save_dir, '0-original')
        setup_dirs(ori_img_dir)
        if denoise:
            den_img_dir = os.path.join(save_dir, '1-denoised')
            setup_dirs(den_img_dir)
            bkg_img_dir = os.path.join(save_dir, '2-background')
            setup_dirs(bkg_img_dir)
            sub_img_dir = os.path.join(save_dir, '3-subtracted')
            setup_dirs(sub_img_dir)
        else:
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
        results = get_ave_intensity(imread(imgf), block=block, offset=offset, denoise=denoise)
        ave_ints.append(results['ave_int'])
        if save_dir is not None:
            imsave(os.path.join(bkg_prof_dir, str(i) + '.png'), results['bkg_prof'])
            imsave(os.path.join(sub_prof_dir, str(i) + '.png'), results['sub_prof'])
            if stylize:
                plt.imsave(os.path.join(ori_img_dir, str(i) + '.png'),
                    results['ori_img'], cmap='viridis')
                if denoise:
                    plt.imsave(os.path.join(den_img_dir, str(i) + '.png'),
                    results['den_img'], cmap='viridis')
                plt.imsave(os.path.join(bkg_img_dir, str(i) + '.png'),
                    results['bkg_img'], cmap='viridis')
                plt.imsave(os.path.join(sub_img_dir, str(i) + '.png'),
                    results['sub_img'], cmap='viridis')
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    imsave(os.path.join(ori_img_dir, str(i) + '.tiff'),
                        img_as_uint(results['ori_img']))
                    if denoise:
                        imsave(os.path.join(den_img_dir, str(i) + '.tiff'),
                            img_as_uint(results['den_img']))
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
        plot(range(len(ave_ints)), np.array(ave_ints)/ave_ints[0],
            xlabel='Frame', ylabel='Fold Change',
            save_path= os.path.join(save_dir, 'fold_change.png'))
        csv_path = os.path.join(save_dir, 'ave_ints.csv')
        np.savetxt(csv_path, np.array(ave_ints), delimiter=',', header='ave_intensity', comments='')
    return ave_ints


def get_regions(img, save_dir=None, gauss_sigma=30, autocontrast=False,
    thres_block=25, thres_offset=-0.005, peaks_min_dist=25, **kwargs):
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
    step_names = ['original', 'denoised', 'subtracted',
        'thresholded', 'peaks', 'regions', 'overlay']
    step_imgs = [img, den, sub, thres, markers, regions, overlay]
    if save_dir is not None:
        for i, (step_name, step_img) in enumerate(zip(step_names, step_imgs)):
            res_dir = os.path.join(save_dir, str(i) + '_' + step_name)
            setup_dirs(res_dir)
            img_path = os.path.join(res_dir, str(len(os.listdir(res_dir))) + '.png')
            if step_name != 'subtracted' or step_name != 'regions':
                plt.imsave(img_path, step_img, cmap='viridis')
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    imsave(img_path, img_as_uint(step_img))
    return dict(zip(step_names, step_imgs))


def images_to_single_cell_trajectories(img_dir, save_dir=None, **reg_params):
    tracker = track.Sort(max_age=3, min_hits=1)
    trajs = defaultdict(defaultdict(dict).copy)
    imgfs = list_img_files(img_dir)
    for i, imgf in enumerate(imgfs):
        img = imread(imgf)
        if save_dir is not None:
            regions_dir = os.path.join(save_dir, 'regions')
            setup_dirs(regions_dir)
            res = get_regions(img, regions_dir, **reg_params)
        else:
            res = get_regions(img, **reg_params)
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
    if save_dir is not None:
        with open(os.path.join(save_dir, 'trajs.json'), 'w') as fp:
            json.dump(trajs, fp)
    return trajs


def filter_trajectories(trajs, min_traj_length=100):
    return {id:info for id, info in trajs.items() if len(info['ints']) > min_traj_length}


def plot_single_cell_trajectories(trajs, save_dir):
    for frames_ints in [info['ints'] for info in trajs.values()]:
        fnums, fints = zip(*frames_ints.items())
        plt.plot(fnums, fints)
    plt.ylabel('Ave Region Intensity')
    plt.xlabel('Frame')
    plt.title('Trajectories')
    fig_name = 'min_' + str(min_traj_length) + '_trajs.png'
    plt.savefig(os.path.join(save_dir, fig_name), bbox_inches='tight')
    plt.close()


def plot_ave_cell_trajectory(trajs, save_dir, by_frame=True):
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


def save_all_traj_data(trajs, save_dir):
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


def run_single_cell_analysis(img_dir,
    save_dir, min_traj_length=100, overwrite=True, **reg_params):
    if overwrite and os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    setup_dirs(save_dir)
    trajs = images_to_single_cell_trajectories(img_dir, save_dir, **reg_params)
    trajs_filtered = filter_trajectories(trajs, min_traj_length)
    plot_single_cell_trajectories(trajs_filtered, save_dir)
    plot_ave_cell_trajectory(trajs, save_dir, by_frame=True)
    save_all_traj_data(trajs_filtered, save_dir)
    save_each_traj_data(trajs, img_dir, save_dir)
