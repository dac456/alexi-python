import getopt, os, sys

import numpy as np
import theano.tensor.nnet as nnet

from mlp import mlp

if not os.path.exists('./networks'):
    os.makedirs('./networks')

load_data, base_name = False, ""
opts, args = getopt.getopt(sys.argv[1:], 'n:')
for o, a in opts:
    if o == '-n':
        load_data = True
        base_name = a

inputs = np.array([[0,1],[1,0],[1,1],[0,0]]).reshape(4,2) #training data X
exp_y = np.array([1, 1, 0, 0]).reshape(4,1) #training data Y

if not load_data:
    model, evaluate = mlp.build_model(2, 10, 1, 10)
    mlp.train_model(model, inputs, exp_y)
    mlp.save_model('networks/' + base_name + '.net', model, evaluate)
else:
    model, evaluate = mlp.load_model('networks/' + base_name + '.net')

print(evaluate([0,1]))
print(evaluate([1,0]))
print(evaluate([1,1]))
print(evaluate([0,0]))
