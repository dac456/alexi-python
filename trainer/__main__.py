import getopt, os, sys, pickle
sys.setrecursionlimit(40000)

import numpy as np
from sklearn import preprocessing
from scipy import stats

import theano
import theano.tensor.nnet as nnet
import theano.tensor as T

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

from mlp import mlp
from mlp import mlp_keras
from preprocessor.parser.image import image_generator
from keras.models import load_model


if not os.path.exists('./networks'):
    os.makedirs('./networks')

base_name, alg = '', ''
rate = 0.1
n_epochs = 500
opts, args = getopt.getopt(sys.argv[1:], 'n:a:r:e:')
for o, a in opts:
    if o == '-n':
        base_name = a
    if o == '-a':
        alg = a
    if o == '-r':
        rate = float(a)
    if o == '-e':
        n_epochs = int(a)


frame_data = None
with open('data/' + base_name + '.framedata.dat', 'rb') as f:
    frame_data = pickle.load(f)
f.close()

image_paths = None
with open('data/' + base_name + '.images.dat', 'rb') as f:
    image_paths = pickle.load(f)
f.close()


inputs = np.asarray(frame_data[:, [3, 4, 5, 6]], dtype=theano.config.floatX)
target = np.asarray(frame_data[:, [2]], dtype=theano.config.floatX)

# model, evaluate, n_batches = mlp.build_model(inputs, target, 2, 5, 1, n_layers=13, batch_size=1, learning_rate=rate, algorithm=alg, activation=T.tanh)
# mlp.train_model(model, n_batches, n_epochs=n_epochs, algorithm=alg)
# mlp.save_model('networks/' + base_name + '.net', model, evaluate)
# model, evaluate = mlp.load_model('networks/' + base_name + '.net')

# model = mlp_keras.build_model(inputs, target, n_layers=13, hidden_dim=7, n_epochs=20, generator=image_generator(base_name, frame_data))
model = mlp_keras.build_model_images(generator=image_generator(base_name, 60000, batch_size=32),
                                     dim=60004, n_samples=inputs.shape[0], hidden_dim=300, n_epochs=5)
model.save('networks/' + base_name + '.h5')
# model = load_model('networks/' + base_name + '.h5')

"""pred = model.predict(inputs)


slope, intercept, r_value, p_value, std_err = stats.linregress(target[:, 0], pred[:, 0])
print("R^2: %f" % r_value)

mse = ((pred - target)**2).mean(axis=None)
print('MSE %f' % mse)

x_t = np.arange(pred.shape[0])
line_exp, = plt.plot(x_t, target, label='expected')
line_pred, = plt.plot(x_t, pred, label='predicted')
plt.legend(handles=[line_exp, line_pred])
plt.show()"""

