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
from jax import random
from jax import vmap
from jax import grad
# from jax.experimental import stax
# from jax.experimental.stax import Dense, Relu, LogSoftmax
import tensorflow_datasets as tfds


def one_hot(x, k, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(k), dtype)

# as_supervised=True gives us the (image, label) as a tuple instead of a dict
ds, info = tfds.load(name='mnist', split='train', as_supervised=True, with_info=True)
# You can build up an arbitrary tf.data input pipeline
ds = ds.batch(128).prefetch(1)
# tfds.dataset_as_numpy converts the tf.data.Dataset into an iterable of NumPy arrays
batches = tfds.as_numpy(ds)
num_labels = info.features['label'].num_classes
h, w, c = info.features['image'].shape
num_pixels = h * w * c


from jax.scipy.special import logsumexp

def relu(x):
  return np.maximum(0, x)

def predict(params, image):
  # per-example predictions
  activations = image
  for w, b in params[:-1]:
    outputs = np.dot(w, activations) + b
    activations = relu(outputs)

  final_w, final_b = params[-1]
  logits = np.dot(final_w, activations) + final_b
  return logits - logsumexp(logits)

def loss(params, images, targets):
  batched_predict = vmap(predict, in_axes=(None, 0))
  preds = batched_predict(params, images)
  return -np.sum(preds * targets)

def f_grad(params):
  x, y = next(batches)
  x = np.reshape(x, (len(x), num_pixels))
  y = one_hot(y, num_labels)
  func = loss(params, x, y)
  func_grad = grad(loss)(params, x, y)
  return func, func_grad


def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

layer_sizes = [784, 512, 512, 10]
params = init_network_params(layer_sizes, random.PRNGKey(0))