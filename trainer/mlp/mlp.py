import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet

def layer(n_in, n_out):
    rng = np.random.RandomState()
    W = np.asarray(
        rng.uniform(
            low=-np.sqrt(6.0 / (n_in + n_out)),
            high=np.sqrt(6.0 / (n_in + n_out)),
            size=(n_in, n_out)
        ),
        dtype=theano.config.floatX
    )

    b = np.zeros((n_out,), dtype=theano.config.floatX)

    return theano.shared(W), theano.shared(b)

def forward(x, W, b, activation):
    return activation(T.dot(x, W) + b)

def backward(theta, cost, learning_rate=0.1):
    return theta - (learning_rate * T.grad(cost, wrt=theta))

def build_model(n_in, n_hidden, n_out, n_layers = 1, activation=T.tanh):
    X = T.dvector('X')
    y = T.dvector('y')

    hidden_layers = []
    updates = []

    i_W, i_b = layer(n_in, n_hidden)
    if n_layers > 1:
        for i in range(n_layers-1):
            h_W, h_b = layer(n_hidden, n_hidden)
            hidden_layers.append((h_W, h_b))
    o_W, o_b = layer(n_hidden, n_out)

    input = forward(X, i_W, i_b, activation)
    last = input
    if n_layers > 1:
        for i, l in enumerate(hidden_layers):
            last = forward(last, l[0], l[1], activation)

    output = forward(last, o_W, o_b, activation)

    cost = T.mean((y - output) ** 2)

    updates.append((o_W, backward(o_W, cost)))
    updates.append((o_b, backward(o_b, cost)))

    for l in reversed(hidden_layers):
        updates.append((l[0], backward(l[0], cost)))
        updates.append((l[1], backward(l[1], cost)))

    updates.append((i_W, backward(i_W, cost)))
    updates.append((i_b, backward(i_b, cost)))


    train = theano.function(inputs=[X,y], outputs=cost, updates=updates)
    evaluate = theano.function(inputs=[X], outputs=output)

    return train, evaluate

def train_model(model, x, y, n_epochs=10000):
    for i in range(n_epochs):
        for j in range(len(x)):
            err = model(x[j], y[j])
        if i % 10 == 0:
            print('Error: %s' % (err,))

    #theano.printing.pydotprint(model, outfile="model.png", var_with_name_simple=True)
