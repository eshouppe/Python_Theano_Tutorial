import gzip
import pickle
import numpy as np
import theano
import theano.tensor as T


class DataProcessing(object):
    """Methods to load data and store in shared variables"""
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        """MNIST dataset has 50k training, 10k validation, and 10k testing images of handwritten
        digits.  Each set is a list of pairs of images and class labels.  Each image is a 1d
        ndarray of (28 x 28) 784.  Labels are between 0 and 9 inclusively."""

        with gzip.open(self.filepath, 'r') as infile:
            training_set, validation_set, testing_set = pickle.load(infile)

        training_x, training_y = self.store_into_shared_vars(training_set)
        validation_x, validation_y = self.store_into_shared_vars(validation_set)
        testing_x, testing_y = self.store_into_shared_vars(testing_set)

        return training_x, training_y, validation_x, validation_y, testing_x, testing_y


    def store_into_shared_vars(self, data_xy):
        """Store datasets into shared variables that will be accessed in minibatches so
        Theano can copy whole dataset to the GPU.  The minibatch is a slice of that shared
        variable.  The minibatch is indicated by its index and size.
        Cast shared_y to int since the values will be used as the batch indices."""

        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))

        return shared_x, T.cast(shared_y, 'int32')


class LogisticRegression(object):
    """Logistic regression model described by weight matrix and bias vector.
    Classify by projecting data points onto a set of hyperplanes, and use
    distance to hyperplane to predict class."""

    def __init__(self, input_minibatch, n_in, n_out):
        """Use shared variables to maintain a persistent state during training. One
        minibatch is described by input, which is of type theano.tensor.TensorType"""

        # Weight matrix W. Column-k represents the separation hyperplane for class-k
        self.W_matrix = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
                                      name='W', borrow=True)
        # Bias vector b.  Element-k represents the free parameter of hyperplane-k
        self.b_vector = theano.shared(value=np.zeros((n_out), dtype=theano.config.floatX),
                                      name='b', borrow=True)

        # Symbolic expression for computing class membership probabilities
        # X is a matrix where row-j represents input training sample-j
        self.p_y_given_x = T.nnet.softmax(T.dot(input_minibatch, self.W_matrix) + self.b_vector)
        # Symbolic description of how to compute predicted class
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # Model parameters
        self.params = [self.W_matrix, self.b_vector]

        # Model input
        self.input_minibatch = input_minibatch

    def negative_log_likelihood(self, y):
        """"Return the mean of the NLL of the model prediction. return var is a
        Theano variable that represents a symbolic expression of the loss. Compile
        the symbolic expression in a Theano function to get the actual value."""

        # Negative mean log likelihood. y.shape[0] is the minibatch size.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the average number of errors in the minibatch"""

        # Make sure y and y_pred have an equal number of dimensions
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )

        if y.dtype.startswith('int'):
            # T.neq returns a vector of 1s/0s where 1 is a mistake
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def train_model(self, learning_rate, cost, batch_size):
        """Gradient descent functions update a parameter set by making small (how small
        determined by learning rate) changes to parameter values that are the gradient
        of a loss function wrt the current parameter values. In stochastic gradient
        descent the update is often performed after each example so the estimate is
        made on a few examples at a time.  In minibatch SGD small batches of training
        examples are used, which reduces variance in the estimate."""
        
        gradient_loss_wrt_W = T.grad(cost=cost, wrt=self.W_matrix)
        gradient_loss_wrt_b = T.grad(cost=cost, wrt=self.b_vector)

        updates = [(self.W_matrix, self.W_matrix - learning_rate*gradient_loss_wrt_W),
                   (self.W_matrix, self.W_matrix - learning_rate*gradient_loss_wrt_W)]

        givens = pass

        train_model = theano.function(
            inputs = [index],
            outputs = cost,
            updates = updates,
            givens = givens
        )
    
        for (x_batch, y_batch) in train_batches:
            print 'Current loss is: ', msgd(x_batch, y_batch)
            if stopping_condition_met:
                return theta


data1 = DataProcessing('mnist.pkl.gz')
train_x, train_y, valid_x, valid_y, test_x, test_y = data1.load_data()

log_reg = LogisticRegression(input_minibatch=train_x, n_in=28*28, n_out=10)
nll_cost = log_reg.negative_log_likelihood(y=train_y)
output = log_reg.train_model(learning_rate=epsilon, cost=nll_cost, batch_size=500)

#######################################################################

# L1 and L2 regularization combat overfitting on training examples by adding
#  a term to the loss function that penalizes certain parameter selections.
# Reglarization terms also penalize large parameter values, which should help
#  find a more general solution.
l1_regularization = T.sum(abs(theta))
l2_sqrd_reglrztn = T.sun(theta ** 2)
# The lambda weight terms are hyperparameters
loss = nll + lambda_1 * l1_regularization + lambda_2 * l2_sqrd_reglrztn

# Save model parameters incrementally during training with pickle
save_file = open('model_parameters', 'w')
pickle.dump(w.get_value(borrow=True), save_file, -1)
pickle.dump(v.get_value(borrow=True), save_file, -1)
pickle.dump(u.get_value(borrow=True), save_file, -1)
save_file.close()
