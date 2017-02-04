import os, sys, getopt, pickle
import numpy as np
import settings
from sklearn import preprocessing
from parser import frame, image


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

if not os.path.exists('./data'):
    os.makedirs('./data')

paths, base_name = [], ''
opts, args = getopt.getopt(sys.argv[1:], 'p:n:')
for o, a in opts:
    if o == '-p':
        paths = a.split(':')
        print("Parsing data in %s" % paths)
    if o == '-n':
        base_name = a


# Get frame data
frame_data = np.concatenate(tuple([frame.parse_frames(path) for path in paths]), axis=0)


# Crop images
# FIXME: this repeats a call to parse_frames for each path
if settings.crop_images:
    for path in paths:
        image.crop(path, frame.parse_frames(path))


# Save image paths
#image_paths = [image.get_image_paths(path) for path in paths]
#with open('data/' + base_name + '.images.dat', 'wb+') as f:
#    pickle.dump(image_paths, f, pickle.HIGHEST_PROTOCOL)
#f.close()

# Save diff images
for path in paths:
    image.parse_images(path, base_name, frame.parse_frames(path))


# Process frame data
if settings.use_scaling:
    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1), copy=True)
    frame_data = scaler.fit_transform(frame_data)

if settings.use_normalization:
    frame_data = preprocessing.normalize(frame_data)

if settings.use_quantization:
    for i in range(frame_data.shape[1]):
        frame_data[:, i] = [quantize_data(x) for x in frame_data[:, i]]

if settings.use_smoothing:
    w = np.ones(250, 'd')
    for i in range(frame_data.shape[1]):
        frame_data[:, i] = np.convolve(w/w.sum(), frame_data[:, i], mode='same')


with open('data/' + base_name + '.framedata.dat', 'wb+') as f:
    pickle.dump(frame_data, f, pickle.HIGHEST_PROTOCOL)
f.close()
