import numpy as np
import theano
import theano.tensor as T

N = 400       # training sample size
FEATS = 784   # number of input variables

# generate a dataset: D = (input_values, target_class)
#
# np.random.randn returns a ndarray of float samples for the
#   standard normal distribution.  Here size is (N x FEATS)
#
# np.random.randint returns a ndarray of random integers in the
#   range low (inclusive) to high (exclusive)
D = (np.random.randn(N, FEATS), np.random.randint(size=N, low=0, high=2))
training_steps = 10000

# Declare Theano symbolic variables
x = T.dmatrix("x")
y = T.dvector("y")

# initialize the weight vector w randomly
# this and the following bias variable b are shared so they keep their
# values between training iterations (updates)
w = theano.shared(np.random.randn(FEATS), name="w")

# initialize the bias term
b = theano.shared(0., name="b")

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))         # probability that target = 1
prediction = p_1 > 0.5                          # prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)   # cross-entropy loss function
cost = xent.mean() + 0.01 * (w**2).sum()        # cost to minimize
# Compute the gradient of the cost wrt weight vector w and bias term b
#   (we shall return to this in a following section of this tutorial)
gw, gb = T.grad(cost, [w, b])

# Compile
train = theano.function(inputs=[x, y], outputs=[prediction, xent],
                        updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = theano.function(inputs=[x], outputs=prediction)

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])
    if i % 1000 == 0:
        print "Latest error: ", sum(err)

print "Final err: ", sum(err)
