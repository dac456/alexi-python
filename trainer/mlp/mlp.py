import numpy as np
import math
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet

import pickle

def rprop_plus_updates(params, grads):

    # RPROP+ parameters
    updates = []
    deltas = 0.1*np.ones(len(params))
    last_weight_changes = np.zeros(len(params))
    last_params = params

    positiveStep = 1.2
    negativeStep = 0.5
    maxStep = 50.
    minStep = math.exp(-6)

    # RPROP+ parameter update (original Reidmiller implementation)
    for param, gparam, last_gparam, delta, last_weight_change in \
            zip(params, grads, last_params, deltas, last_weight_changes):
        # calculate change
        change = T.sgn(gparam * last_gparam)
        if T.gt(change, 0) :
            delta = T.minimum(delta * positiveStep, maxStep)
            weight_change = T.sgn(gparam) * delta
            last_gparam = gparam

        elif T.lt(change, 0):
            delta = T.maximum(delta * negativeStep, minStep)
            weight_change = -last_weight_change
            last_gparam = 0

        else:
            weight_change = T.sgn(gparam) * delta
            last_gparam = param

        # update the weights
        updates.append((param, param - weight_change))
        # store old change
        last_weight_change = weight_change

    return updates

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

def forward(x, W, b, activation):
    return activation(T.dot(x, W) + b)

#def backward(theta, cost, learning_rate=0.01):
#    return theta - (learning_rate * T.grad(cost, wrt=theta))

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

    #last = forward(X, params[0], params[1], activation)
    #last = activation(T.dot(X, params[0]) + params[1])
    #for i in range(2, len(params), 2):
        #last = forward(last, params[i], params[i+1], activation)
    #    last = activation(T.dot(last, params[i]) + params[i+1])
    #output = last
    out = X
    for i in range(0, len(params), 2):
        out = activation(T.dot(out, params[i]) + params[i+1])

    ssq = T.sum([T.sum(params[i]**2) for i in range(0, len(params), 2)])

    L2 = 0.001 * ssq
    cost = T.mean((y - out) ** 2) + L2

    gparams = [T.grad(cost, param) for param in params]

    if algorithm == 'rprop':
        updates = rprop_plus_updates(params, gparams)
    else:
        updates = [(param, param - learning_rate * gparam) for param, gparam in zip(params, gparams)]

    if algorithm == 'batch' or algorithm == 'rprop':
        train = theano.function(inputs=[idx], outputs=cost, updates=updates, givens={
            X: data_in[idx * batch_size: (idx+1) * batch_size],
            y: data_target[idx * batch_size: (idx+1) * batch_size]
        })
        #train = theano.function(inputs=[X,y], outputs=cost, updates=updates)
    evaluate = theano.function(inputs=[X], outputs=out)

    return train, evaluate, input.shape[0] // batch_size

def train_model(model, n_batches, n_epochs=10000, algorithm='incremental'):
    if algorithm == 'batch' or algorithm == 'rprop':
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
