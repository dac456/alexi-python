import os
import configparser
import numpy as np
from .util import numerical_sort


def parse_frames(path):
    out_array = np.array([]).reshape(0, 9)

    files = []
    for (dirpath, dirnames, filenames) in os.walk(os.path.abspath(path) + '/framedata'):
        files.extend(filenames)
        break

    files = sorted(files, key=numerical_sort)

    for f in files:
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

        # if warped:
        #    print('debug: dx %f dy %f' % dx, dy)


        out_array = np.append(out_array, [[dx,dy,dtheta,vleft,vright,pitch,roll,vx,vy]], axis=0)

    return out_array
