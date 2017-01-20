import os, re
import configparser
import numpy as np

# http://stackoverflow.com/questions/12093940/reading-files-in-a-particular-order-in-python
numbers = re.compile(r'(\d+)')
def numerical_sort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def parse_frames(path):
    out_array = np.array([]).reshape(0, 7)

    files = []
    for (dirpath, dirnames, filenames) in os.walk(path + '/framedata'):
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
        dx = cfg.getfloat('global', 'vdxr')
        dy = cfg.getfloat('global', 'vdyr')
        dtheta = cfg.getfloat('global', 'vdtheta')
        vleft = cfg.getfloat('global', 'vleft')
        vright = cfg.getfloat('global', 'vright')
        pitch = cfg.getfloat('global', 'vpitch')
        roll = cfg.getfloat('global', 'vroll')

        out_array = np.append(out_array, [[dx,dy,dtheta,vleft,vright,pitch,roll]], axis=0)

    return out_array
