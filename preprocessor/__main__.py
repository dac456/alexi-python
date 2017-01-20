import os, sys, getopt, pickle
import numpy as np
from sklearn import preprocessing
from parser import frame

if not os.path.exists('./data'):
    os.makedirs('./data')

path, base_name = '', ''
scale = False
opts, args = getopt.getopt(sys.argv[1:], 'p:sn:')
for o, a in opts:
    if o == '-p':
        path = a
        print("Parsing data in %s" % path)
    if o == '-s':
        scale = True
        print("Scaling data")
    if o == '-n':
        base_name = a


frame_data = frame.parse_frames(path)
if scale:
    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1), copy=True)
    #frame_data = preprocessing.scale(frame_data, axis=1, with_mean=True, with_std=True, copy=True)
    frame_data = scaler.fit_transform(frame_data)
    #frame_data = preprocessing.normalize(frame_data, axis=0)

with open('data/' + base_name + '.framedata.dat', 'wb+') as f:
    pickle.dump(frame_data, f, pickle.HIGHEST_PROTOCOL)
f.close()
