import numpy as np
from mlp import mlp
import theano.tensor.nnet as nnet


inputs = np.array([[0,1],[1,0],[1,1],[0,0]]).reshape(4,2) #training data X
exp_y = np.array([1, 1, 0, 0]).reshape(4,1) #training data Y

model, evaluate = mlp.build_model(2, 10, 1, 10)

mlp.train_model(model, inputs, exp_y)

print(evaluate([0,1]))
print(evaluate([1,0]))
print(evaluate([1,1]))
print(evaluate([0,0]))
