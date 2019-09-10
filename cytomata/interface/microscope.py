import os
import time
import warnings
from collections import defaultdict, deque

import numpy as np
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
        self.settings = settings
        self.save_dir = settings['save_dir']
        setup_dirs(self.save_dir)
        with open(os.path.join(self.save_dir, 'settings.json'), 'w') as fp:
            json.dump(settings, fp)
        self.img_w = settings['pixel_size'] * settings['img_width']
        self.img_h = settings['pixel_size'] * settings['img_height']
        self.core.setExposure(settings['cam_exposure'])
        self.core.setProperty('Camera', 'Gain', settings['cam_gain'])
        self.tasks = []
        setup_dirs(self.save_dir, 'tasks_log')
        self.xt = defaultdict(list)
        self.xx = defaultdict(list)
        self.xy = defaultdict(list)
        self.xz = defaultdict(list)
        self.uta = []
        self.utb = []
        self.av = []
        self.az = []
        self.x0 = self.get_position('x')
        self.y0 = self.get_position('y')
        self.z0 = self.get_position('z')
        self.coords = np.array([[
            self.x0,
            self.y0,
            self.z0
        ]])
        self.t0 = time.time()

    def queue_induction(self, t_info, ch_ind, mag):
        for (start, stop, period, width) in t_info:
            times = deque(np.arange(start + self.t0, stop + self.t0, period))
            self.tasks.append({
                'func': self.pulse_light,
                'times': times,
                'kwargs': {'width': width, 'ch_ind': ch_ind, 'mag': mag}
            })
        t = time.strftime('%Y%m%d-%H%M%S')
        with open(os.path.join(self.save_dir, 'tasks_log', t + '-induction.json'), 'w') as fp:
            json.dump({'t_info': t_info, 'ch_ind': ch_ind, 'mag': mag}, fp)

    def queue_imaging(self, t_info, chs):
        for (start, stop, period) in t_info:
            times = deque(np.arange(start + self.t0, stop + self.t0, period))
            self.tasks.append({
                'func': self.image_coords,
                'times': times,
                'kwargs': {'chs': chs}
            })
        for ch in chs:
            for i in range(len(self.coords)):
                setup_dirs(os.path.join(self.save_dir, ch, str(i)))
        t = time.strftime('%Y%m%d-%H%M%S')
        with open(os.path.join(self.save_dir, 'tasks_log', t + '-imaging.json'), 'w') as fp:
            json.dump({'t_info': t_info, 'chs': chs}, fp)

    def queue_autofocus(self, t_info, ch):
        for (start, stop, period) in t_info:
            times = deque(np.arange(start + self.t0, stop + self.t0, period))
            self.tasks.append({
                'func': self.autofocus,
                'times': times,
                'kwargs': {'ch': ch}
            })
        t = time.strftime('%Y%m%d-%H%M%S')
        with open(os.path.join(self.save_dir, 'tasks_log', t + '-autofocus.json'), 'w') as fp:
            json.dump({'t_info': t_info, 'ch': ch}, fp)

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

    def set_channel(self, chname):
        if chname != self.core.getCurrentConfig('Channel'):
            self.core.setConfig('Channel', chname)

    def set_magnification(self, mag):
        if mag != self.core.getState('TINosePiece'):
            self.core.setState('TINosePiece', mag)

    def get_position(self, axis):
        if axis.lower() == 'x':
            return self.core.getXPosition('XYStage')
        elif axis.lower() == 'y':
            return self.core.getYPosition('XYStage')
        elif axis.lower() == 'z':
            return self.core.getPosition('TIZDrive')
        else:
            raise ValueError('Invalid axis arg in Microscope.get_position(axis).')

    def set_position(self, axis, value):
        x0 = self.get_position('x')
        y0 = self.get_position('y')
        z0 = self.get_position('z')
        if axis.lower() == 'xy':
            self.core.setXYPosition('XYStage', value[0], value[1])
        elif axis.lower() == 'z':
            self.core.setPosition('TIZDrive', value)
        else:
            raise ValueError('Invalid axis arg in Microscope.set_position(axis).')

    def shift_position(self, axis, value):
        if axis.lower() == 'xy':
            self.core.setRelativeXYPosition('XYStage', value[0], value[1])
        elif axis.lower() == 'z':
            self.core.setRelativePosition('TIZDrive', value)
        else:
            raise ValueError('Invalid axis arg in Microscope.shift_position(axis).')

    def add_coord(self):
        x = self.get_position('x')
        y = self.get_position('y')
        z = self.get_position('z')
        self.coords = np.vstack((self.coords, [x, y, z]))

    def snap_image(self):
        self.core.waitForSystem()
        self.core.snapImage()
        return self.core.getImage()

    def add_coords_session(self):
        self.core.setAutoShutter(0)
        self.core.setShutterOpen(1)
        while True:
            ans = raw_input('Enter [y] to add current (x, y, z) to coord list or any key to quit.')
            if ans.lower() == 'y':
                self.add_coord()
                print('--Coords List--')
                for coord in self.coords:
                    print(coord)
            else:
                break
        self.core.setShutterOpen(0)
        self.core.setAutoShutter(1)

    def snap_zstack(self, bounds, step):
        z0 = self.coords[0, 2]
        zi = self.get_position('z')
        zl = np.max([zi + bounds[0], z0 - 50.0])
        zu = np.min([zi + bounds[1], z0 + 50.0])
        positions = list(np.arange(zl, zu, step))
        imgs = []
        for z in positions:
            self.set_position('z', z)
            img = self.snap_image()
            imgs.append(img)
            ch = self.core.getCurrentConfig('Channel')
            tstamp = time.strftime('%Y%m%d-%H%M%S')
            img_path = os.path.join(self.save_dir,
                'zstack_' + tstamp, ch, str(z) + '.tiff')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imsave(img_path, img)
        self.set_position('z', zi)
        return positions, imgs

    def snap_xyfield(self, n=5, step=81.92):
        x0 = self.get_position('x')
        y0 = self.get_position('y')
        xs = np.arange(-(n//2)*step, (n//2)*step + step, step)
        ys = np.arange(-(n//2)*step, (n//2)*step + step, step)
        for i, yi in enumerate(ys):
            for xi in xs:
                if not i % 2:
                    xi = -xi
                self.set_position('xy', (x0 + xi, y0 + yi))
                img = self.snap_image()
                ch = self.core.getCurrentConfig('Channel')
                tstamp = time.strftime('%Y%m%d-%H%M%S')
                save_path = os.path.join(self.save_dir,
                    'xyfield_' + tstamp, ch, str(xi) + '-' + str(yi) + '.tiff')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    imsave(img_path, img)
        self.set_position('xy', (x0, y0))

    def measure_focus(self):
        img = self.snap_image()
        return np.var(laplace(img))

    def pulse_light(self, width, ch_ind, mag):
        mag0 = self.core.getState('TINosePiece')
        self.set_magnification(mag)
        ta = time.time() - self.t0
        self.set_channel(ch_ind)
        time.sleep(width)
        tb = time.time() - self.t0
        self.set_channel('None')
        self.set_magnification(mag0)
        self.uta.append(ta)
        self.utb.append(tb)
        u_path = os.path.join(self.save_dir, 'u' + str(i) + '.csv')
        u_data = np.column_stack((self.uta, self.utb))
        np.savetxt(u_path, u_data, delimiter=',', header='u_ta,u_tb', comments='')

    def image_coords(self, chs):
        for i, (x, y, z) in enumerate(self.coords):
            self.set_position('xy', (x, y))
            self.set_position('z', z)
            self.xt[i].append(time.time() - self.t0)
            self.xx[i].append(x)
            self.xy[i].append(y)
            self.xz[i].append(z)
            for ch in chs:
                self.set_channel(ch)
                img = self.snap_image()
                img_name = str(round(self.xt[i][-1]))
                img_path = os.path.join(self.save_dir, ch, str(i), img_name + '.tiff')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    imsave(img_path, img)
            x_path = os.path.join(self.save_dir, 'x' + str(i) + '.csv')
            x_data = np.column_stack((self.xt[i], self.xx[i], self.xy[i], self.xz[i]))
            np.savetxt(x_path, x_data, delimiter=',', header='t,x,y,z', comments='')

    def autofocus(self, ch='DIC', algo='brent', bounds=[-3.0, 3.0], max_iter=5, offset=0):
        self.set_channel(ch)
        if algo == 'brent':  # Brent's Method
            def residual(z):
                self.set_position('z', z)
                foc = measure_focus()
                return -foc
            zi = self.get_position('z')
            zl = np.max([zi + bounds[0], self.z0 - 50.0])
            zu = np.min([zi + bounds[1], self.z0 + 50.0])
            result = minimize_scalar(residual, method='bounded',
                bounds=(zl, zu), options={'maxiter': max_iter, 'xatol': 2.0})
            best_pos, best_foc = result.x, -result.fun
        else:  # Reset Position
            best_pos = self.z0
            best_foc = 0.0
        self.coords[:, 2] += (best_pos - self.z0) + offset  # Update all z-coords
        self.az.append(best_pos)
        self.av.append(best_foc)
        a_path = os.path.join(self.save_dir, 'a.csv')
        a_data = np.column_stack((self.az, self.av))
        np.savetxt(a_path, a_data, delimiter=',', header='af_z,af_v', comments='')
