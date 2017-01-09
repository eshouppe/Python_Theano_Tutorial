import gzip
import pickle
import numpy as np
import theano
import theano.tensor as T


def load_data(filepath):
    """MNIST dataset has 50k training, 10k validation, and 10k testing images of handwritten
    digits.  Each set is a list of pairs of images and class labels.  Each image is a 1d
    ndarray of (28 x 28) 784.  Labels are between 0 and 9 inclusively."""

    with gzip.open(filepath, 'r') as f:
        training_set, validation_set, testing_set = pickle.load(f)

    return training_set, validation_set, testing_set


def store_into_shared_vars(data_xy):
    """Store datasets into shared variables that will be accessed in minibatches so
    Theano can copy whole dataset to the GPU.  The minibatch is a slice of that shared
    variable.  The minibatch is indicated by its index and size.
    Cast shared_y to int since the values will be used as the batch indices."""

    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))

    return shared_x, T.cast(shared_y, 'int32')


""""Both of the following loss objectives are Theano variables that
represent a symbolic expression of the loss. Compile the symbolic
expression in a Theano function to get the actual value."""

# Zero-one loss counts the number of errors, i.e. where the true y and k from 
#  argmax(P(Y=k|x,theta)) do not equal
zero_one_loss = T.sum(T.neq(T.argmax(p_y_given_x), y))

# Negative log likelihood is a differentiable loss function that is minimized
nll = -T.sum(T.log(p_y_given_x)[T.arange(y.shape[0]), y])

"""Gradient descent functions update a parameter set by making small (how small
determined by learning rate) changes to parameter values that are the derivative 
(the gradient) of a loss function wrt the current parameter set values. In stochastic
gradient descent the update is often performed after each example or over minibatches."""


train_set, valid_set, test_set = load_data('mnist.pkl.gz')
train_x, train_y = store_into_shared_vars(train_set)
valid_x, valid_y = store_into_shared_vars(valid_set)
test_x, test_y = store_into_shared_vars(test_set)

BATCH_SIZE = 500
