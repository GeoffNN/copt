"""
Combining COPT with JAX
=======================

This example shows how JAX can be used within COPT
to compute the gradients of the objective function.
In this example tensorflow-datasets is used to provide
the training data.
"""
import copt as cp
import jax.numpy as np
import numpy as onp
from jax import random
from jax import vmap
from jax import grad
import pylab as plt

# .. construct (random) dataset ..
n_samples, n_features = 1000, 200
key = random.PRNGKey(1)
X = random.normal(key, (n_samples, n_features))
key, subkey = random.split(key)
y = random.normal(key, (n_samples,))


def loss(w):
  # squared error loss
  z = X.dot(w) - y
  return np.sum(z * z)

def f_grad(w):
  return loss(w), grad(loss)(w)

w0 = onp.zeros(n_features)

l1_ball = cp.utils.L1Ball(n_features / 2.)
cb = cp.utils.Trace(loss)
cp.minimize_proximal_gradient(
    f_grad,
    w0,
    verbose=True,
    callback=cb
)
plt.plot(cb.trace_fx)
plt.xlabel('# Iterations')
plt.ylabel('Objective value')
plt.grid()
plt.show()