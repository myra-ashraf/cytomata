import os
import time
import warnings
from collections import defaultdict, deque

import numpy as np
import cv2
from skimage.io import imsave
from skimage.filters import laplace
from scipy.optimize import minimize_scalar

import MMCorePy
from cytomata.utils.io import setup_dirs


class Microscope(object):
    """
    Microscope task automation and data recording.
    """
    def __init__(self, settings, config_file):
        self.core = MMCorePy.CMMCore()
        self.core.loadSystemConfiguration(config_file)
        self.tasks = []
        self.settings = settings
        self.save_dir = settings['save_dir']
        self.ch_group = self.settings['ch_group']
        self.obj_device = self.settings['obj_device']
        self.xy_device = self.settings['xy_device']
        self.z_device = self.settings['z_device']
        self.img_w = self.settings['img_width_um']
        self.img_h = self.settings['img_height_um']
        for dev in settings['img_sync']:
            self.core.assignImageSynchro(dev)
        self.uta = defaultdict(list)
        self.utb = defaultdict(list)
        self.x0 = self.get_position('x')
        self.y0 = self.get_position('y')
        self.z0 = self.get_position('z')
        self.xlim = np.array(settings['stage_x_limit']) + self.x0
        self.ylim = np.array(settings['stage_y_limit']) + self.y0
        self.zlim = np.array(settings['stage_z_limit']) + self.z0
        self.coords = np.array([[self.x0, self.y0, self.z0]])
        self.cid = 0
        self.t0 = time.time()

    def set_channel(self, chname):
        if chname != self.core.getCurrentConfig(self.ch_group):
            self.core.setConfig(self.ch_group, chname)

    def set_magnification(self, mag):
        if mag != self.core.getState(self.obj_device):
            self.core.setState(self.obj_device, mag)

    def get_position(self, axis):
        if axis.lower() == 'x' and self.xy_device:
            return self.core.getXPosition(self.xy_device)
        elif axis.lower() == 'y' and self.xy_device:
            return self.core.getYPosition(self.xy_device)
        elif axis.lower() == 'z' and self.z_device:
            return self.core.getPosition(self.z_device)
        else:
            raise ValueError('Invalid axis arg in Microscope.get_position(axis).')

    def set_position(self, axis, value):
        if axis.lower() == 'xy' and self.xy_device:
            if (value[0] > self.xlim[0] and value[0] < self.xlim[1] and
            value[1] > self.ylim[0] and value[1] < self.ylim[1]):
                self.core.setXYPosition(self.xy_device, value[0], value[1])
        elif axis.lower() == 'z' and self.z_device:
            if value > self.zlim[0] and value < self.zlim[1]:
                self.core.setPosition(self.z_device, value)
        else:
            raise ValueError('Invalid axis arg in Microscope.set_position(axis).')

    def add_coord(self):
        x = self.get_position('x')
        y = self.get_position('y')
        z = self.get_position('z')
        self.coords = np.vstack((self.coords, [x, y, z]))

    def add_coords_session(self, ch):
        self.set_channel(ch)
        cv2.namedWindow('Coordinate Picker')
        self.core.startContinuousSequenceAcquisition(1)
        while True:
            if self.core.getRemainingImageCount() > 0:
                img = self.core.getLastImage()
                img = cv2.normalize(img, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
                cv2.imshow('Coordinate Picker', img)
            k = cv2.waitKey(20)
            if k == 27:  # ESC - Exit
                break
            elif k == 32:  # SPACE - Add Current Coord
                self.add_coord()
                print('--Coords List--')
                for coord in self.coords:
                    print(coord)
            elif k == 8:  # Backspace - Remove Prev Coord
                self.coords = self.coords[:-1]
                print('--Coords List--')
                for coord in self.coords:
                    print(coord)
        cv2.destroyAllWindows()
        self.core.stopSequenceAcquisition()

    def snap_image(self):
        self.core.snapImage()
        return self.core.getImage()

    def snap_zstack(self, chs, zdepth, step):
        z0 = self.get_position('z')
        zs = np.arange(zi - zdepth/2, zi + zdepth/2, step)
        tstamp = time.strftime('%Y%m%d-%H%M%S')
        img_dir = os.path.join(self.save_dir, tstamp + '_zstack')
        imgs = []
        for ch in chs:
            self.set_channel(ch)
            for z in zs:
                self.set_position('z', z)
                img = self.snap_image()
                imgs.append((z, ch, img))
        for z, ch, img in imgs:
            ch_dir = os.path.join(img_dir, ch)
            setup_dirs(ch_dir)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imsave(os.path.join(ch_dir, str(z) + '.tiff'), img)
        self.set_position('z', z0)

    def snap_xyfield(self, chs, n=5, step=81.92):
        x0 = self.get_position('x')
        y0 = self.get_position('y')
        grid = np.arange(-(n//2)*step, (n//2)*step + step, step)
        tstamp = time.strftime('%Y%m%d-%H%M%S')
        img_dir = os.path.join(self.save_dir, tstamp + '_xyfield')
        imgs = []
        for ch in chs:
            self.set_channel(ch)
            for i, yi in enumerate(grid):
                for j, xi in enumerate(grid):
                    if not i % 2:
                        xi = -xi
                    self.set_position('xy', (x0 + xi, y0 + yi))
                    img = self.snap_image()
                    imgs.append((i, j, ch, img))
        for i, j, ch, img in imgs:
            ch_dir = os.path.join(img_dir, ch)
            setup_dirs(ch_dir)
            img_path = os.path.join(ch_dir, str(i) + '_' + str(j) + '.tiff')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imsave(img_path, img)
        self.set_position('xy', (x0, y0))

    def take_images(self, cid, chs):
        (x, y, z) = self.coords[cid]
        self.set_position('xy', (x, y))
        ti = time.time() - self.t0
        for ch in chs:
            self.set_channel(ch)
            img = self.snap_image()
            img_path = os.path.join(self.save_dir, ch, str(cid), str(round(ti, 1)) + '.tiff')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imsave(img_path, img)

    def imaging_task(self, chs):
        if self.settings['mpos']:
            if self.settings['mpos_mode'] == 'parallel':
                for i in range(len(self.coords)):
                    self.take_images(i, chs)
            elif self.settings['mpos_mode'] == 'sequential':
                self.take_images(self.cid, chs)
        else:
            self.take_images(self.cid, chs)

    def queue_imaging(self, t_info, chs):
        for (start, stop, period) in t_info:
            times = deque(np.arange(start + self.t0, stop + self.t0, period))
            self.tasks.append({
                'func': self.imaging_task,
                'times': times,
                'kwargs': {'chs': chs}
            })
        for ch in chs:
            for i in range(len(self.coords)):
                setup_dirs(os.path.join(self.save_dir, ch, str(i)))

    def pulse_light(self, cid, width, ch_ind):
        (x, y, z) = self.coords[cid]
        self.set_position('xy', (x, y))
        ch0 = self.core.getCurrentConfig(self.ch_group)
        ta = time.time() - self.t0
        self.set_channel(ch_ind)
        time.sleep(width)
        tb = time.time() - self.t0
        self.set_channel(ch0)
        self.uta[cid].append(ta)
        self.utb[cid].append(tb)
        u_path = os.path.join(self.save_dir, 'u' + str(cid) + '.csv')
        u_data = np.column_stack((self.uta[cid], self.utb[cid]))
        np.savetxt(u_path, u_data, delimiter=',', header='ta,tb', comments='')

    def induction_task(self, width, ch_ind):
        if self.settings['mpos']:
            if self.settings['mpos_mode'] == 'parallel':
                for i in range(len(self.coords)):
                    self.pulse_light(i, width, ch_ind)
            elif self.settings['mpos_mode'] == 'sequential':
                self.pulse_light(self.cid, width, ch_ind)
        else:
            self.pulse_light(self.cid, width, ch_ind)

    def queue_induction(self, t_info, ch_ind):
        for (start, stop, period, width) in t_info:
            times = deque(np.arange(start + self.t0, stop + self.t0, period))
            self.tasks.append({
                'func': self.induction_task,
                'times': times,
                'kwargs': {'width': width, 'ch_ind': ch_ind}
            })

    def measure_focus(self):
        img = self.snap_image()
        return np.var(laplace(img))

    def autofocus(self, cid, ch, bounds=[-3.0, 3.0], max_iter=5, offset=0):
        self.set_channel(ch)
        (x, y, z) = self.coords[cid]
        self.set_position('xy', (x, y))
        zi = self.get_position('z')
        zl = np.max([zi + bounds[0], self.z0 - 50.0])
        zu = np.min([zi + bounds[1], self.z0 + 50.0])
        def residual(z):
            self.set_position('z', z)
            foc = self.measure_focus()
            return -foc
        result = minimize_scalar(residual, method='bounded',
            bounds=(zl, zu), options={'maxiter': max_iter, 'xatol': 2.0})
        best_pos, best_foc = result.x, -result.fun
        self.coords[cid] = best_pos + offset

    def autofocus_task(self, ch, bounds, max_iter, offset):
        if self.settings['mpos']:
            if self.settings['mpos_mode'] == 'parallel':
                for i in range(len(self.coords)):
                    self.autofocus(i, ch, bounds, max_iter, offset)
            elif self.settings['mpos_mode'] == 'sequential':
                self.autofocus(self.cid, ch, bounds, max_iter, offset)
        else:
            self.autofocus(self.cid, ch, bounds, max_iter, offset)

    def queue_autofocus(self, t_info, ch):
        for (start, stop, period) in t_info:
            times = deque(np.arange(start + self.t0, stop + self.t0, period))
            self.tasks.append({
                'func': self.autofocus_task,
                'times': times,
                'kwargs': {'ch': ch}
            })

    def run_tasks(self):
        if self.tasks:
            for i, task in enumerate(self.tasks):
                if task['times']:
                    if time.time() > task['times'][0]:
                        task['func'](**task['kwargs'])
                        self.tasks[i]['times'].popleft()
                else:
                    self.tasks.pop(i)
        else:
            return True
