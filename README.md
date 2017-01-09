### Python Theano Tutorial

This repo contains scratch code to complete the exercises in the Python Theano basic tutorial at http://deeplearning.net/software/theano/tutorial/index.html#.
The basic tutorial includes examples for derivatives, conditions, and loops.  But importantly it also covers shared variables and the paradigm for using Theano.

With Theano, we first initialize symbolic variables of some type.  Often we then define an expression graph which connects the variables according to their symbolic function.  And finally we then compile the graph with a theano function.

Using logistic regressions as an example, first declare the symbolic variables.
 ```python
 import theano.tensor as T

 x = T.dmatrix("x")
 y = T.dvector("y") 
 ```
 x and y are instances of **TensorVariable** and are assigned types dmatrix and dvector respectively, where d is double (float64).

The full expression graph can be found in logistic_regression.py.  However one part of the graph is the prediction computation.  In binary logistic regression the probability of 1, which is often notated as p or pi, is the logit of the odds.  Solving for p, this equals
```python
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))
```
After the full expression graph is defined, the Theano function is compiled.
```python
train = theano.function(inputs=[x, y],
                        outputs=[prediction, xent],
                        updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
``` 
Then we train the function some large number of times to tune the parameters, which are coefficients of the explanatory variables in this example.  

---

University of Montreal LISA Lab Theano deep learning tutorial
http://deeplearning.net/tutorial/