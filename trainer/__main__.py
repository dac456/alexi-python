import getopt, os, sys, pickle
sys.setrecursionlimit(40000)

import numpy as np
import theano.tensor.nnet as nnet
import theano.tensor as T
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

from mlp import mlp

if not os.path.exists('./networks'):
    os.makedirs('./networks')

base_name, alg = '', ''
rate = 0.1
opts, args = getopt.getopt(sys.argv[1:], 'n:a:r:')
for o, a in opts:
    if o == '-n':
        base_name = a
    if o == '-a':
        alg = a
    if o == '-r':
        rate = float(a)

#inputs = np.array([[0,1],[1,0],[1,1],[0,0]]).reshape(4,2) #training data X
#exp_y = np.array([1, 1, 0, 0]).reshape(4,1) #training data Y

frame_data = None
with open('data/' + base_name + '.framedata.dat', 'rb') as f:
    frame_data = pickle.load(f)
f.close()

inputs = frame_data[:,[3,4]]
target = frame_data[:,[2]]
print(inputs.shape)

w=np.ones(99,'d')
target = np.convolve(w/w.sum(),target[:,0],mode='same').reshape(target.shape[0],1)
inputs[:,0] = np.convolve(w/w.sum(),inputs[:,0],mode='same')
inputs[:,1] = np.convolve(w/w.sum(),inputs[:,1],mode='same')
print(inputs)

plt.plot(inputs[:,0])
plt.show()

model, evaluate, n_batches = mlp.build_model(inputs, target, 2, 100, 1, n_layers=5, batch_size=100, learning_rate=rate, algorithm=alg, activation=T.tanh)
mlp.train_model(model, n_batches, n_epochs=2000, algorithm=alg)
mlp.save_model('networks/' + base_name + '.net', model, evaluate)
#model, evaluate = mlp.load_model('networks/' + base_name + '.net')

pred = evaluate(inputs)

mse = ((pred - target)**2).mean(axis=None)
print(mse)

print(target)
print(pred)

plt.plot(pred)
plt.show()
