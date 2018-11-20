import os
import sys
import pickle
from collections import defaultdict
sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.io import imread
from skimage.measure import regionprops

from cytomata.process import track
from cytomata.utils.visual import plot
from cytomata.utils.io import list_fnames_sorted, setup_dirs
from cytomata.process.extract import get_ave_intensity, get_regions


def images_to_ave_frame_intensities(img_dir, out_path=None):
    ave_ints = []
    for fn in list_fnames_sorted(img_dir):
        img = imread(os.path.join(img_dir, fn))
        ave_int = get_ave_intensity(img)
        ave_ints.append(ave_int)
    ave_ints = np.array(ave_ints)
    if out_path is not None:
        np.savetxt(out_path, ave_ints, delimiter=',', header='t', comments='')
    return ave_ints


def images_to_ave_single_cell_intensities(img_dir, out_dir=None, params={}):
    tracker = track.Sort(max_age=3, min_hits=1)
    trajs = defaultdict(defaultdict(dict).copy)
    for i, fn in enumerate(list_fnames_sorted(img_dir)):
        img = imread(os.path.join(img_dir, fn))
        regions_dir = os.path.join(out_dir, 'regions')
        setup_dirs(regions_dir)
        regions, img_processed = get_regions(img, regions_dir, **params)
        ints = np.array([prop.mean_intensity for prop in regionprops(regions, img_processed)])
        dets = np.array([prop.bbox for prop in regionprops(regions)])
        trks = tracker.update(dets)
        for trk in trks:  # Restructure data trajectory-wise
            id = int(trk[4])
            ind = np.argmax(np.array([track.iou(det, trk) for det in dets]))
            if len(trajs[id]['color']) == 3:
                ecolor = trajs[id]['color']
            else:
                ecolor = np.random.rand(3,)
            trajs[id]['color'] = ecolor
            trajs[id]['dets'][i] = dets[ind]
            trajs[id]['trks'][i] = trk[0:4]
            trajs[id]['ints'][i] = ints[ind]
    # with open(os.path.join(out_dir, 'trajs.pkl'), 'wb') as fp:
    #     pickle.dump(trajs, fp)
    return trajs


def plot_single_cell_trajectories(trajs, out_dir, min_traj_length=100):
    trajs = {id:info for id, info in trajs.items() if len(info['ints']) > min_traj_length}
    trajs_frames_ints = [info['ints'] for info in trajs.values()]
    for frames_ints in trajs_frames_ints:
        fnums, fints = zip(*frames_ints.items())
        plt.plot(fnums, fints)
    plt.ylabel('Ave Region Intensity')
    plt.xlabel('Frame')
    plt.title('Trajectories')
    fig_name = 'min_' + str(min_traj_length) + '_trajs.png'
    plt.savefig(os.path.join(out_dir, fig_name), bbox_inches='tight')
    plt.close()


def plot_single_cell_trajectories_ave(trajs, out_dir, by_frame=True):
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
    plt.savefig(os.path.join(out_dir, 'trajs_frame_aves.png'), bbox_inches='tight')
    plt.close()


def save_single_cell_data(img_dir, trajs, out_dir, min_traj_length=100):
    trajs = {id:info for id, info in trajs.items() if len(info['ints']) > min_traj_length}
    img_names = list_fnames_sorted(img_dir)
    for id, info in trajs.items():
        id_dir = os.path.join(out_dir, 'traj', str(id))
        setup_dirs(id_dir)
        rec_frames = []
        rec_flints = []
        for tframe, tbbox in info['trks'].items():
            flint = info['ints'][tframe]
            rec_frames.append(tframe)
            rec_flints.append(flint)
            fig, ax = plt.subplots()
            ax.imshow(imread(os.path.join(img_dir, img_names[tframe])))
            ax.axis('off')
            rect = Rectangle((tbbox[1], tbbox[0]), tbbox[3] - tbbox[1],
                tbbox[2] - tbbox[0], fill=False, edgecolor=info['color'], linewidth=1)
            ax.add_patch(rect)
            ax.set_title('Frame: ' + str(tframe) + ' | ' + 'Intensity: ' + str(round(flint, 4)))
            fig.savefig(os.path.join(id_dir, str(tframe) + '.png'), bbox_inches='tight')
            plt.close(fig)
        traj_path = os.path.join(id_dir, 'traj_int.png')
        plot(rec_frames, rec_flints, labels=None, xlabel='Frame',
            ylabel='Ave Intensity', title='Trajectory #' + str(id), save_path=traj_path)
        flint_path = os.path.join(id_dir, 'flint.csv')
        np.savetxt(flint_path, np.column_stack((rec_frames, rec_flints)), delimiter=',', header='frame,fl_int', comments='')


if __name__ == '__main__':
    data_dir = 'data'
    expt = 'linshan'
    # ch = '1'
    # img_dir = os.path.join(data_dir, expt, ch)
    # # out_path = os.path.join(data_dir, 'ave_frame_results.csv')
    # out_dir = os.path.join(data_dir, expt + '_' + ch + '_' + 'single_cell_results')
    # setup_dirs(out_dir)
    # # images_to_ave_frame_intensities(img_dir, out_path)
    # trajs = images_to_ave_single_cell_intensities(img_dir, out_dir)
    # plot_single_cell_trajectories(trajs, out_dir)
    # plot_single_cell_trajectories_ave(trajs, out_dir)
    # save_single_cell_data(img_dir, trajs, out_dir)

    ch = '2'
    img_dir = os.path.join(data_dir, expt, ch)
    out_dir = os.path.join(data_dir, expt + '_' + ch + '_' + 'single_cell_results')
    setup_dirs(out_dir)
    trajs = images_to_ave_single_cell_intensities(img_dir, out_dir)
    plot_single_cell_trajectories(trajs, out_dir)
    plot_single_cell_trajectories_ave(trajs, out_dir)
    save_single_cell_data(img_dir, trajs, out_dir)
