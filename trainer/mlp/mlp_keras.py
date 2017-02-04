from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import SGD
from keras.regularizers import l2, activity_l2
from keras.layers.advanced_activations import LeakyReLU
from keras.metrics import cosine_proximity
import keras.backend as K


def r2_score(y_true, y_pred):
    ss_res = K.sum(K.pow(y_true - y_pred, 2))
    ss_tot = K.sum(K.pow(y_true - K.mean(y_true), 2))

    return 1.0 - (ss_res/ss_tot)


def build_model(x, y, n_layers=1, hidden_dim=20, n_epochs=20):
    n_in = x.shape[1]
    n_out = y.shape[1]

    m = Sequential()
    m.add(Dense(hidden_dim, input_dim=n_in, init='uniform', W_regularizer=l2(0.001)))
    m.add(BatchNormalization())
    m.add(Activation('tanh'))
    #m.add(Dropout(0.25))

    for i in range(n_layers-1):
        m.add(Dense(hidden_dim, init='uniform', W_regularizer=l2(0.001)))
        m.add(BatchNormalization())
        m.add(Activation('tanh'))
        #m.add(Dropout(0.25))

    #m.add(Dropout(0.1))
    m.add(Dense(n_out, init='uniform', W_regularizer=l2(0.001)))
    m.add(BatchNormalization())
    m.add(Activation('tanh'))

    m.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

    m.fit(x, y, nb_epoch=n_epochs, batch_size=32, validation_split=0.1)

    return m


def build_model_images(generator, dim, n_samples, n_layers=10, hidden_dim=200, n_epochs=20):
    m = Sequential()
    m.add(Dense(hidden_dim, input_dim=dim, init='uniform', W_regularizer=l2(0.001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))

    for i in range(n_layers-1):
        m.add(Dense(hidden_dim, init='uniform', W_regularizer=l2(0.001)))
        m.add(BatchNormalization())
        m.add(Activation('relu'))

    m.add(Dense(dim - 4, init='uniform', W_regularizer=l2(0.001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))

    m.compile(loss='mse', optimizer='adam', metrics=[r2_score])

    m.fit_generator(generator, samples_per_epoch=n_samples, nb_epoch=n_epochs)

    return m
