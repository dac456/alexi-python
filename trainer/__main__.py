import argparse, os, sys, pickle
sys.setrecursionlimit(40000)

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
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

from preprocessor import settings


if not os.path.exists('./networks'):
    os.makedirs('./networks')

parser = argparse.ArgumentParser(description="neural net trainer")
parser.add_argument('-n', dest='base_name')
parser.add_argument('--terrain', dest='mode', action='store_true')
args = parser.parse_args()

frame_data = None
with open('data/' + args.base_name + '.framedata.dat', 'rb') as f:
    frame_data = pickle.load(f)

test_data = None
with open('data/' + args.base_name + '.testdata.dat', 'rb') as f:
    test_data = pickle.load(f)

val_data = None
with open('data/' + args.base_name + '.valdata.dat', 'rb') as f:
    val_data = pickle.load(f)

inputs = np.asarray(frame_data[:, [3, 4, 5, 6]], dtype=theano.config.floatX)
dx = np.asarray(frame_data[:, [0]], dtype=theano.config.floatX)
dy = np.asarray(frame_data[:, [1]], dtype=theano.config.floatX)
speed = np.asarray(frame_data[:, [2]], dtype=theano.config.floatX)
dtheta = np.asarray(frame_data[:, [3]], dtype=theano.config.floatX)

inputs_test = np.asarray(test_data[:, [3, 4, 5, 6]])
dx_test = np.asarray(test_data[:, [0]])
dy_test = np.asarray(test_data[:, [1]])
speed_test = np.asarray(test_data[:, [2]])
dtheta_test = np.asarray(test_data[:, [3]])

"""inputs_val = np.asarray(val_data[:, [3, 4, 5, 6]])
dx_val = np.asarray(val_data[:, [0]])
dy_val = np.asarray(val_data[:, [1]])
dtheta_val = np.asarray(val_data[:, [2]])
speed_val = np.sqrt(dx_val**2 + dy_val**2)"""

img_dim = settings.crop_dimensions[0] * settings.crop_dimensions[1]

if not args.mode:
    pass

    #dx_model = mlp_keras.build_model(inputs, dx, n_layers=13, hidden_dim=7, n_epochs=25)
    #dx_model.save(f'networks/{args.base_name}.dx.h5')

    #dy_model = mlp_keras.build_model(inputs, dy, n_layers=13, hidden_dim=7, n_epochs=25)
    #dy_model.save(f'networks/{args.base_name}.dy.h5')

    #speed_model = mlp_keras.build_model(inputs, speed, n_layers=13, hidden_dim=7, n_epochs=25)
    #speed_model.save(f'networks/{args.base_name}.speed.h5')

    #dtheta_model = mlp_keras.build_model(inputs, dtheta, n_layers=13, hidden_dim=7, n_epochs=25)
    #dtheta_model.save(f'networks/{args.base_name}.dtheta.h5')
else:
    terrain_model = mlp_keras.build_model_images(generator=image_generator(args.base_name, img_dim, batch_size=32),
                                                 dim=img_dim, n_samples=inputs.shape[0], hidden_dim=2000,
                                                 n_layers=10, n_epochs=5)
    terrain_model.save(f'networks/{args.base_name}.terrain.h5')


terrain_model = load_model(f'networks/{args.base_name}.terrain.h5')
X = np.empty((0, 128*128 + 4))
Y = np.empty((0, 128*128))
idx = 0
for x, y in image_generator(args.base_name, 128*128, batch_size=1, num_samples=8000):
    print(f'idx: {idx}')
    X = np.append(X, x, axis=0)
    Y = np.append(Y, y, axis=0)
    idx += 1
    if idx == 8000:
        break
print(X.shape)
Y_pred = terrain_model.predict(X)
print(f'r2: {r2_score(Y, Y_pred, multioutput="uniform_average")}')
print(f'mse: {mean_squared_error(Y, Y_pred, multioutput="uniform_average")}')


#dx_model = load_model(f'networks/{args.base_name}.dx.h5')
#dy_model = load_model(f'networks/{args.base_name}.dy.h5')
speed_model = load_model(f'networks/{args.base_name}.speed.h5')
dtheta_model = load_model(f'networks/{args.base_name}.dtheta.h5')


#pred_dx = dx_model.predict(inputs_test)
#mse_dx = ((pred_dx - dx_test)**2).mean(axis=None)
#slope, intercept, r_value_dx, p_value, std_err = stats.linregress(dx_test[:, 0], pred_dx[:, 0])
#print(f'dx R^2: {r_value_dx} MSE: {mse_dx}')

#pred_dy = dy_model.predict(inputs_test)
#mse_dy = ((pred_dy - dy_test)**2).mean(axis=None)
#slope, intercept, r_value_dy, p_value, std_err = stats.linregress(dy_test[:, 0], pred_dy[:, 0])
#print(f'dy R^2: {r_value_dy} MSE: {mse_dy}')

pred_speed = speed_model.predict(inputs_test)
mse_speed = ((pred_speed - speed_test)**2).mean(axis=None)
slope, intercept, r_value_speed, p_value, std_err = stats.linregress(speed_test[:, 0], pred_speed[:, 0])
print(f'speed R^2: {r_value_speed} MSE: {mse_speed}')

pred_dtheta = dtheta_model.predict(inputs_test)
mse_dtheta = ((pred_dtheta - dtheta_test)**2).mean(axis=None)
slope, intercept, r_value_dtheta, p_value, std_err = stats.linregress(dtheta_test[:, 0], pred_dtheta[:, 0])
print(f'dtheta R^2: {r_value_dtheta} MSE: {mse_dtheta}')

"""vl = inputs_test[:, [0]].reshape(inputs_test.shape[0],)
vr = inputs_test[:, [1]].reshape(inputs_test.shape[0],)
wd = np.abs(vl - vr)
rho_v_real = stats.pearsonr(wd, speed_test[:, 0])
rho_v_sim = stats.pearsonr(wd, pred_speed[:, 0])
print(f'rho real: {rho_v_real}, rho sim: {rho_v_sim}')
rho_v_real = stats.pearsonr(wd, dtheta_test[:, 0])
rho_v_sim = stats.pearsonr(wd, pred_dtheta[:, 0])
print(f'rho real: {rho_v_real}, rho sim: {rho_v_sim}')"""


#plt.title(rf'$\Delta x$ Real vs Predicted on Test Set ($MSE = {np.around(mse_dx, 2)}$)')
#x_t = np.arange(dx_test.shape[0])
#real, = plt.plot(x_t, dx_test)
#pred, = plt.plot(x_t, pred_dx)
#plt.legend([real, pred], ['Real', 'Predicted'])
#plt.show()

#plt.title(rf'$\Delta y$ Real vs Predicted on Test Set ($MSE = {np.around(mse_dy, 2)}$)')
#x_t = np.arange(dy_test.shape[0])
#real, = plt.plot(x_t, dy_test)
#pred, = plt.plot(x_t, pred_dy)
#plt.legend([real, pred], ['Real', 'Predicted'])
#plt.show()

plt.title(rf'$v$ Real vs Predicted on Test Set ($MSE = {np.around(mse_speed, 4)}$)')
x_t = np.arange(dy_test.shape[0])
real, = plt.plot(x_t, speed_test)
pred, = plt.plot(x_t, pred_speed)
plt.legend([real, pred], ['Real', 'Predicted'])
plt.show()

plt.title(rf'$\Delta \theta$ Real vs Predicted on Test Set ($MSE = {np.around(mse_dtheta, 4)}$)')
x_t = np.arange(dtheta_test.shape[0])
real, = plt.plot(x_t, dtheta_test)
pred, = plt.plot(x_t, pred_dtheta)
plt.legend([real, pred], ['Real', 'Predicted'])
plt.show()


#plt.title(rf'$\Delta x$ Real vs Predicted on Test Set ($R^2 = {np.around(r_value_dx, 2)}$)')
#plt.xlabel('Real Value')
#plt.ylabel('Predicted Value')
#plt.scatter(dx_test, pred_dx, marker='.', s=1)
#fit = np.polyfit(dx_test.reshape(dx_test.shape[0]), pred_dx.reshape(pred_dx.shape[0]), deg=1)
#plt.plot(dx_test, fit[0] * dx_test + fit[1], color='red')

#plt.show()

#plt.title(rf'$\Delta y$ Real vs Predicted on Test Set ($R^2 = {np.around(r_value_dy, 2)}$)')
#plt.xlabel('Real Value')
#plt.ylabel('Predicted Value')
#plt.scatter(dy_test, pred_dy, marker='.', s=1)
#fit = np.polyfit(dy_test.reshape(dy_test.shape[0]), pred_dy.reshape(pred_dy.shape[0]), deg=1)
#plt.plot(dy_test, fit[0] * dy_test + fit[1], color='red')

#plt.show()

plt.title(rf'$v$ Real vs Predicted on Test Set ($R^2 = {np.around(r_value_speed, 4)}$)')
plt.xlabel('Real Value')
plt.ylabel('Predicted Value')
plt.scatter(speed_test, pred_speed, marker='.', s=1)
fit = np.polyfit(speed_test.reshape(speed_test.shape[0]), pred_speed.reshape(pred_speed.shape[0]), deg=1)
plt.plot(speed_test, fit[0] * speed_test + fit[1], color='red')

plt.show()

plt.title(rf'$\Delta \theta$ Real vs Predicted on Test Set ($R^2 = {np.around(r_value_dtheta, 4)}$)')
plt.xlabel('Real Value')
plt.ylabel('Predicted Value')
plt.scatter(dtheta_test, pred_dtheta, marker='.', s=1)
fit = np.polyfit(dtheta_test.reshape(dtheta_test.shape[0]), pred_dtheta.reshape(pred_dtheta.shape[0]), deg=1)
plt.plot(dtheta_test, fit[0] * dtheta_test + fit[1], color='red')

plt.show()
