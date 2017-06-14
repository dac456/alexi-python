from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import r2_score
import keras.backend as K

def r2_score(y_true, y_pred):
    SS_res = K.sum(K.pow(y_true - y_pred, 2.0))
    SS_tot = K.sum(K.pow(y_true - K.mean(y_true), 2.0))

    return 1.0 - (SS_res / SS_tot)

class Callbacks(Callback):
    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.model.validation_data[0])
        r2 = r2_score(self.model.validation_data[1], y_pred)
        print(f'R^2: {r2}')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def build_model(x, y, n_layers=1, hidden_dim=20, n_epochs=20, val_input=None, val_target=None, l2_param=0.0001):
    n_in = x.shape[1]
    n_out = y.shape[1]

    m = Sequential()
    # m.add(Dropout(0.05, input_shape=(n_in,)))
    m.add(Dense(hidden_dim, input_dim=n_in, init='uniform', W_regularizer=l2(l2_param)))
    m.add(BatchNormalization())
    m.add(Activation('tanh'))

    for i in range(n_layers-1):
        m.add(Dense(hidden_dim, init='uniform', W_regularizer=l2(l2_param)))
        m.add(BatchNormalization())
        m.add(Activation('tanh'))

    m.add(Dense(n_out, init='uniform', W_regularizer=l2(l2_param)))
    m.add(BatchNormalization())
    m.add(Activation('tanh'))

    m.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

    m.fit(x, y, nb_epoch=n_epochs, batch_size=32)

    return m


def build_model_images(generator, dim, n_samples, n_layers=10, hidden_dim=200, n_epochs=20):
    m = Sequential()
    m.add(Dense(hidden_dim, input_dim=dim + 4, init='glorot_uniform', W_regularizer=l2(0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('tanh'))

    for i in range(n_layers-1):
        m.add(Dense(hidden_dim, init='glorot_uniform', W_regularizer=l2(0.0001)))
        m.add(BatchNormalization())
        m.add(Activation('tanh'))

    m.add(Dense(dim, init='glorot_uniform', W_regularizer=l2(0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('tanh'))

    m.compile(loss='mse', optimizer='adam')

    m.fit_generator(generator, samples_per_epoch=n_samples, nb_epoch=n_epochs)

    return m
