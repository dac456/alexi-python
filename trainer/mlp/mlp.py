import numpy as np
import math
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet

from mlp import rprop

import pickle


def layer(n_in, n_out, rng):
    W = np.asarray(
        rng.uniform(
            low=-np.sqrt(6.0 / (n_in + n_out)),
            high=np.sqrt(6.0 / (n_in + n_out)),
            size=(n_in, n_out)
        ),
        dtype=theano.config.floatX
    )

    b = np.zeros((n_out,), dtype=theano.config.floatX)

    return [theano.shared(W), theano.shared(b)]


def build_model(input, target, n_in, n_hidden, n_out, n_layers = 1, batch_size=20, learning_rate = 0.1, activation=T.tanh, algorithm='incremental'):
    X = T.matrix('X')
    y = T.matrix('y')
    idx = T.lscalar()

    data_in = theano.shared(np.asarray(input, dtype=theano.config.floatX), borrow=True)
    data_target = theano.shared(np.asarray(target, dtype=theano.config.floatX), borrow=True)

    params = []
    updates = []

    rng = np.random.RandomState()
    params = layer(n_in, n_hidden, rng)
    for i in range(n_layers-1):
        params =  params + layer(n_hidden, n_hidden, rng)
    params = params + layer(n_hidden, n_out, rng)

    out = X
    for i in range(0, len(params)-2, 2):
        out = activation(T.dot(out, params[i]) + params[i+1])
    L = len(params)-2
    out = T.dot(out, params[L]) + params[L+1]

    L1 = 0.00001 * T.sum([T.sum(abs(params[i])) for i in range(0, len(params), 2)])
    L2 = 0.0001 * T.sum([T.sum(params[i]**2) for i in range(0, len(params), 2)])
    cost = T.mean((y - out) ** 2) + L2

    gparams = [T.grad(cost, param) for param in params]

    if algorithm == 'rprop-':
        updates = rprop.irprop_minus_updates(params, gparams)
    elif algorithm == 'rprop+':
        updates = rprop.rprop_plus_updates(params, gparams)
    else:
        updates = [(param, param - learning_rate * gparam) for param, gparam in zip(params, gparams)]

    if 'batch' in algorithm or 'rprop' in algorithm:
        train = theano.function(inputs=[idx], outputs=cost, updates=updates, givens={
            X: data_in[idx * batch_size: (idx+1) * batch_size],
            y: data_target[idx * batch_size: (idx+1) * batch_size]
        })
        #train = theano.function(inputs=[X,y], outputs=cost, updates=updates)
    evaluate = theano.function(inputs=[X], outputs=out)

    return train, evaluate, input.shape[0] // batch_size

def train_model(model, n_batches, n_epochs=10000, algorithm='incremental'):
    if 'batch' in algorithm or 'rprop' in algorithm:
        epoch = 0
        while (epoch < n_epochs):
            epoch = epoch + 1
            for idx in range(n_batches):
                err = model(idx)
            if epoch % 10 == 0:
                print('Error: %s' % (err,))
    else:
        print('Unknown algorithm: %s' % algorithm)

    #theano.printing.pydotprint(model, outfile="model.png", var_with_name_simple=True)

def save_model(file, model, evaluate):
    with open(file, 'wb+') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(evaluate, f, pickle.HIGHEST_PROTOCOL)
    f.close()

    print('Network saved as: %s' % file)


def load_model(file):
    with open(file, 'rb') as f:
        fn = []
        for i in range(2):
            fn.append(pickle.load(f))
    f.close()

    return fn[0], fn[1]
