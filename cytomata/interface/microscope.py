import os
import time
import copy
import warnings
from collections import defaultdict, deque

import numpy as np
from skimage.io import imsave
from skimage.filters import laplace
from scipy.optimize import minimize_scalar

import MMCorePy
from cytomata.utils.io import setup_dirs


dir_path = os.path.dirname(os.path.realpath(__file__))


class Microscope(object):
    """
    Microscope task automation and data recording.
    """
    def __init__(self, save_dir, tasks, config_file=os.path.join(dir_path, 'mm.cfg')):
        self.core = MMCorePy.CMMCore()
        self.core.loadSystemConfiguration(config_file)
        # self.core.assignImageSynchro('XYStage')
        # self.core.assignImageSynchro('TIZDrive')
        self.save_dir = save_dir
        self.tasks = tasks
        self.events = []
        self.funcs = {
            'pulse_light': self.pulse_light,
            'image_coords': self.image_coords,
            'autofocus': self.autofocus
        }
        self.xt = defaultdict(list)
        self.xx = defaultdict(list)
        self.xy = defaultdict(list)
        self.xz = defaultdict(list)
        self.ut = defaultdict(list)
        self.av = []
        self.az = []
        self.coords = np.array([[
            self.get_position('x'),
            self.get_position('y'),
            self.get_position('z')
        ]])
        while True:
            ans = raw_input('Add current (x, y, z) to coords list? y/[n]: ')
            if ans.lower() == 'y':
                self.add_coord()
            else:
                break
        if 'imaging' in self.tasks:
            for ch in self.tasks['imaging']['kwargs']['chs']:
                for i in range(len(self.coords)):
                    setup_dirs(os.path.join(self.save_dir, ch, str(i)))
        self.t0 = time.time()
        self.queue_init_events()

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
        print('Current Pos: ({0}, {1}, {2})'.format(x0, y0, z0))
        print('Move To: {0}={1}'.format(axis, value))
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
        # self.core.waitForSystem()
        self.core.snapImage()
        return self.core.getImage()

    def snap_zstack(self, ch, bounds, step, save=True):
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
            if save:
                img_path = os.path.join(self.save_dir, 'zstack', ch, str(z) + '.tiff')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    imsave(img_path, img)
        self.set_position('z', zi)
        return positions, imgs

    # def snap_xyfield(self, ch, n=5, step=81.92):
    #     x0 = self.get_position('x')
    #     y0 = self.get_position('y')
    #     xs = np.arange(-(n//2)*step, (n//2)*step + step, step)
    #     ys = np.arange(-(n//2)*step, (n//2)*step + step, step)
    #     for i, yi in enumerate(ys):
    #         for xi in xs:
    #             if not i % 2:
    #                 xi = -xi
    #             self.set_position('xy', (x0 + xi, y0 + yi))
    #             img = self.snap_image()
    #             save_path = os.path.join(self.save_dir, 'xyfield', ch,
    #                 str(xi) + '-' + str(yi), time.strftime('%H%M%S') + '.tiff')
    #             with warnings.catch_warnings():
    #                 warnings.simplefilter("ignore")
    #                 imsave(img_path, img)
    #     self.set_position('xy', (x0, y0))

    def measure_focus(self, img):
        return np.var(laplace(img))

    def sample_focus(self):
        pos = self.get_position('z')
        foc = self.measure_focus(self.snap_image())
        return pos, foc

    def sample_focus_stack(self, bounds=[-3.0, 3.0], step=1.0):
        positions, imgs = self.snap_zstack(ch='DIC', bounds=bounds, step=step, save=False)
        focuses = [self.measure_focus(img) for img in imgs]
        return positions, focuses

    def queue_event(self, func, tstart, tstop, tstep, kwargs):
        times = deque(np.arange(tstart + self.t0, tstop + self.t0, tstep))
        self.events.append({'func': func, 'times': times, 'kwargs': kwargs})

    def queue_init_events(self):
        for task, params in self.tasks.items():
            n_events = len(params['t_starts'])
            for i in range(n_events):
                kwargs = copy.deepcopy(params['kwargs'])
                for k, v in kwargs.items():
                    if isinstance(v, list) and len(v) == n_events:
                        kwargs[k] = kwargs[k][i]
                self.queue_event(
                    func=self.funcs[params['func']],
                    tstart=params['t_starts'][i],
                    tstop=params['t_stops'][i],
                    tstep=params['t_steps'][i],
                    kwargs=kwargs,
                )

    def run_events(self):
        if self.events:
            for i, event in enumerate(self.events):
                if event['times']:
                    if time.time() > event['times'][0]:
                        event['func'](**event['kwargs'])
                        self.events[i]['times'].popleft()
                else:
                    self.events.pop(i)
        else:
            return True

    def pulse_light(self, ch_ind, ch_dark, width):
        for i, (x, y, z) in enumerate(self.coords):
            self.set_position('xy', (x, y))
            self.set_position('z', z)
            t1 = time.time() - self.t0
            self.set_channel(ch_ind)
            time.sleep(width)
            t2 = time.time() - self.t0
            self.set_channel(ch_dark)
            self.ut[i] += [t1, t2]
            u_path = os.path.join(self.save_dir, 'u' + str(i) + '.csv')
            np.savetxt(u_path, self.ut[i], delimiter=',', header='ut', comments='')

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

    def autofocus(self, ch='DIC', algo='brent', bounds=[-3.0, 3.0], max_iter=5):
        self.set_channel(ch)
        z0 = self.coords[0, 2]
        if algo == 'grid':  # Grid Search
            positions, focuses = self.sample_focus_stack(bounds=bounds, num=max_iter)
            best_foc = max(focuses)
            best_pos = positions[focuses.index(max(focuses))]
        elif algo == 'brent':  # Brent's Method
            def residual(z):
                self.set_position('z', z)
                foc = self.sample_focus()[1]
                print('z: ' + str(z))
                print('f: ' + str(foc))
                return -foc
            zi = self.get_position('z')
            zl = np.max([zi + bounds[0], z0 - 10.0])
            zu = np.min([zi + bounds[1], z0 + 50.0])
            print('--autofocus--')
            result = minimize_scalar(residual, method='bounded',
                bounds=(zl, zu), options={'maxiter': max_iter, 'xatol': 2.0})
            best_pos, best_foc = result.x, -result.fun
        else:  # Reset Position
            best_pos = z0
            best_foc = 0.0
        self.coords[:, 2] += (best_pos - self.coords[0, 2])  # Update all z-coords
        self.az.append(best_pos)
        self.av.append(best_foc)
        a_path = os.path.join(self.save_dir, 'a.csv')
        a_data = np.column_stack((self.az, self.av))
        np.savetxt(a_path, a_data, delimiter=',', header='af_z, af_v', comments='')
