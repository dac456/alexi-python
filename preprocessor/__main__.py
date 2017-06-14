import os, sys, getopt, pickle
import numpy as np
import settings
from sklearn import preprocessing
from parser import frame, image

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt


if not os.path.exists('./data'):
    os.makedirs('./data')

paths, base_name, test_path, val_path = [], '', '', ''
process_images = True # TODO: get from args
opts, args = getopt.getopt(sys.argv[1:], 'p:n:t:v:')
for o, a in opts:
    if o == '-p':
        paths = a.split(':')
        print("Parsing data in %s" % paths)
    if o == '-t':
        test_path = a
    if o == '-v':
        val_path = a
    if o == '-n':
        base_name = a


# Get training data
all_data = np.empty((0, 10))
all_raw = np.empty((0, 10))
excluded = []
slices = []
last_end = 0
for idx, path in enumerate(paths):
    f_raw = frame.parse_frames(path)
    all_raw = np.append(all_raw, f_raw[0], axis=0)
    excluded += list(f_raw[1])
    frame_data = np.asarray([x for idx, x in enumerate(f_raw[0]) if idx not in f_raw[1]])
    all_data = np.append(all_data, frame_data, axis=0)

    slices.append((last_end, all_raw.shape[0]))
    last_end = all_raw.shape[0]

outliers = frame.reject_outliers(all_data, m=3)
all_data = np.asarray([x for idx, x in enumerate(all_data) if idx not in set(outliers)])

# Save min-max scalers before applying all transforms
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=True)
scaler.fit(all_data[:, [4, 5, 6, 7]])
with open(f'data/{base_name}.framedata.scaler_in.dat', 'wb+') as f:
    pickle.dump(scaler, f, pickle.HIGHEST_PROTOCOL)

scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=True)
scaler.fit(all_data[:, [0, 1, 2, 3]])
with open(f'data/{base_name}.framedata.scaler_out.dat', 'wb+') as f:
    pickle.dump(scaler, f, pickle.HIGHEST_PROTOCOL)

all_data = frame.apply_transforms(all_data)

all_excluded = outliers + excluded

if process_images:
    for idx, path in enumerate(paths):
        exclude = [x for x in all_excluded if x < slices[idx][1]]
        image_processor = image.ImageProcessor(base_name, path, all_raw[slices[idx][0]:slices[idx][1], :],
                                               exclude)
        image_processor.crop()
        image_processor.parse()


with open(f'data/{base_name}.framedata.dat', 'wb+') as f:
    pickle.dump(all_data, f, pickle.HIGHEST_PROTOCOL)

with open(f'data/{base_name}.framedata_img.dat', 'wb+') as f:
    for row in all_data:
        pickle.dump(row, f, pickle.HIGHEST_PROTOCOL)

# Get test data
f_raw = frame.parse_frames(test_path)

all_data = np.empty((0, 10))
all_raw = np.empty((0, 10))
excluded = []
all_raw = np.append(all_raw, f_raw[0], axis=0)
excluded += list(f_raw[1])
frame_data = np.asarray([x for idx, x in enumerate(f_raw[0]) if idx not in f_raw[1]])
all_data = np.append(all_data, frame_data, axis=0)

outliers = frame.reject_outliers(all_data, m=3)
all_data = np.asarray([x for idx, x in enumerate(all_data) if idx not in set(outliers)])
#test_data = np.asarray([x for idx, x in enumerate(f_raw[0]) if idx not in f_raw[1]])
test_data = frame.apply_transforms(all_data)

with open(f'data/{base_name}.testdata.dat', 'wb+') as f:
    pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)
    f.close()

# Get validation data
"""f_raw = frame.parse_frames(val_path)
val_data = np.asarray([x for idx, x in enumerate(f_raw[0]) if idx not in f_raw[1]])
val_data = frame.apply_transforms(val_data)

with open(f'data/{base_name}.valdata.dat', 'wb+') as f:
    pickle.dump(val_data, f, pickle.HIGHEST_PROTOCOL)
    f.close()"""
