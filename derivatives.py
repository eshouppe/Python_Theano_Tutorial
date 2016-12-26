import numpy as np
import theano
import theano.tensor as T


# The derivative or gradient of an expression y wrt x
# The grad function works symbolically.  It receives & returns Theano variables.
x = T.dscalar('x')
y = x ** 2
gy = T.grad(y, x)
print theano.pp(gy) # Confirm that the gradient

# The Theano Jacobian is the tensor that comprises the first partial
#   derivatives of the output of a function wrt its inputs
x = T.dvector('x')
y = x ** 2
# scan is a generic operation that allows recurrent equations to be
#   written in a symbolic manner
J, updates = theano.scan(lambda i, y, x: T.grad(y[i], x),
                         sequences=T.arange(y.shape[0]),
                         non_sequences=[y, x])
f = theano.function([x], J, updates=updates)
print f([4, 4])

# The Theano Hessian is the matrix compromising the second order partial
#   derivatives of a function with scalar output and vector input.
x = T.dvector('x')
y = x ** 2
cost = y.sum()
gy = T.grad(cost, x)
H, updates = theano.scan(lambda i, gy, x: T.grad(gy[i], x),
                         sequences=T.arange(gy.shape[0]),
                         non_sequences=[gy, x])
print f([4, 4])
