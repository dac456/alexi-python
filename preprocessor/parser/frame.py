import os
import configparser
import numpy as np
from sklearn import preprocessing
from preprocessor import settings
from .util import numerical_sort


def quantize_data(x):
    y = 0
    if x > 0:
        if 0.2 > x >= 0:
            y = 0.0
        elif 0.4 > x >= 0.2:
            y = 0.2
        elif 0.6 > x >= 0.4:
            y = 0.4
        elif 0.8 > x >= 0.6:
            y = 0.6
        elif 1.0 > x >= 0.8:
            y = 0.8

    elif x < 0:
        if -0.2 < x < 0:
            y = 0.0
        elif -0.4 < x <= -0.2:
            y = -0.2
        elif -0.6 < x <= -0.4:
            y = -0.4
        elif -0.8 < x <= -0.6:
            y = -0.6
        elif -1.0 < x <= -0.8:
            y = -0.8

    return y


def parse_frames(path, swap_xy=False):
    out_array = np.array([]).reshape(0, 10)
    excluded = []

    files = []
    for (dirpath, dirnames, filenames) in os.walk(os.path.abspath(path) + '/framedata'):
        files.extend(filenames)
        break

    files = sorted(files, key=numerical_sort)

    for idx, f in enumerate(files):
        cfg = configparser.RawConfigParser()

        # http://stackoverflow.com/questions/2819696/parsing-properties-file-in-python/25493615#25493615
        # framedata files generated by chrono don't have sections
        with open(path + '/framedata/' + f, 'r') as cfg_file:
            config_string = '[global]\n' + cfg_file.read()

        cfg.read_string(config_string)
        warped = cfg.getboolean('global', 'warped')

        dx = cfg.getfloat('global', 'vdxr')
        dy = cfg.getfloat('global', 'vdyr')
        dtheta = cfg.getfloat('global', 'vdtheta')
        vleft = cfg.getfloat('global', 'vleft')
        vright = cfg.getfloat('global', 'vright')
        pitch = cfg.getfloat('global', 'vpitch')
        roll = cfg.getfloat('global', 'vroll')

        vx = cfg.getint('global', 'vx')
        vy = cfg.getint('global', 'vy')

        if swap_xy:
            dx, dy = dy, dx

        speed = np.sqrt(dx**2 + dy**2)

        # if not warped:
        out_array = np.append(out_array, [[dx,dy,speed,dtheta,vleft,vright,pitch,roll,vx,vy]], axis=0)
        if warped:
            excluded.append(idx)

    return out_array, set(excluded)


def apply_transforms(data, add_noise=False):
    y = data

    if settings.use_scaling:
        print('applying scaling...')
        y[:, :8] = preprocessing.minmax_scale(y[:, :8], feature_range=(-1, 1), axis=0)

    if settings.use_smoothing:
        print('applying smoothing...')
        w = np.ones(250, 'd')
        for i in range(8):
            y[:, i] = np.convolve(w / w.sum(), y[:, i], mode='same')

    if settings.use_quantization:
        print('applying quantization...')
        for i in range(8):
            y[:, i] = [quantize_data(x) for x in y[:, i]]

    return y


# NOTE: needs to be run on entire dataset
def reject_outliers(data, m):
    excluded = []

    dx_mean = np.mean(data[:, [0]])
    dx_std = np.std(data[:, [0]])
    dy_mean = np.mean(data[:, [1]])
    dy_std = np.std(data[:, [1]])
    for idx, row in enumerate(data):
        if np.abs(row[0] - dx_mean) >= m * dx_std:
            excluded.append(idx)
        if np.abs(row[1] - dy_mean) >= m * dy_std:
            excluded.append(idx)

    return excluded
