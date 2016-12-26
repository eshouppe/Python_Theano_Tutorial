"""Theano tutorial with Python 2.7"""

import numpy as np
import theano
import theano.tensor as T
from theano import pp


# T.dscalar is a Theano type for 0-dimensional arrays of doubles
# T.dmatrix is a Theano type for matrices of doubles

# Calling a tensor function with a string argument creates a variable
#  representing the desired quantity with the given name

# Basic logistic curve: s(x) = 1 / [1 + (e^-x)].  
# Logistic function may be applied elementwise to a matrix

# Theano can use default values for arguments that can be 
#   modified positionally or by name
w, x , y = T.dscalars('w', 'x', 'y')
z = (x + y) * w
f = theano.function([x, theano.In(y, value=3), theano.In(w, value=2, name='w_by_name')], z)

print f(33) # = 72
print f(33, 2) # = 70
print f(33, w_by_name=1) # = 36

# Shared variables are hybrid symbolic and non-symbolic variables whose value
#  may be shared between multiple functions.  They may be used in symbolic 
#  expressions but they also have an internal value that defines the value taken
#  by the symbolic variable in all functions that use it.
state = theano.shared(0)
inc = T.iscalar('inc')
accumulator = theano.function([inc], state, updates=[(state, state+inc)])
decrementor = theano.function([inc], state, updates=[(state, state-inc)])
print state.get_value() # = 0
accumulator(1)
print state.get_value() # = 1
decrementor(4)
print state.get_value() # = -3
state.set_value(-10)
accumulator(70)
print state.get_value() # = 60

# Theano functions can be copied so that another function can be similar but
#  with different variables or updates
new_state = theano.shared(0)
new_accumulator = accumulator.copy(swap={state: new_state})
new_accumulator(100)
print new_state.get_value() # = 100
print state.get_value() # still = 60

# In Theano you first express everything symbolically and then compile theano
#  expressions to get functions.

