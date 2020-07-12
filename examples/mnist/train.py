# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MNIST example.

This script trains a simple Convolutional Neural Net on the MNIST dataset.
The data is loaded using tensorflow_datasets.

"""

from absl import app
from absl import flags
from absl import logging

import functools

from flax import jax_utils
from flax import nn
from flax import optim
from flax.metrics import tensorboard

import jax
from jax import random

import jax.numpy as jnp

import numpy as onp

import tensorflow_datasets as tfds


FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate', default=0.1,
    help=('The learning rate for the momentum optimizer.'))

flags.DEFINE_float(
    'momentum', default=0.9,
    help=('The decay rate used for the momentum optimizer.'))

flags.DEFINE_integer(
    'batch_size', default=128,
    help=('Batch size for training.'))

flags.DEFINE_integer(
    'num_epochs', default=10,
    help=('Number of training epochs.'))

flags.DEFINE_string(
    'model_dir', default=None,
    help=('Directory to store model data.'))


class CNN(nn.Module):
  """A simple CNN model."""

  def apply(self, x):
    x = nn.Conv(x, features=32, kernel_size=(3, 3))
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(x, features=64, kernel_size=(3, 3))
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(x, features=256)
    x = nn.relu(x)
    x = nn.Dense(x, features=10)
    x = nn.log_softmax(x)
    return x


@jax.pmap
def create_model(key):
  _, initial_params = CNN.init_by_shape(key, [((1, 28, 28, 1), jnp.float32)])
  model = nn.Model(CNN, initial_params)
  return model


@functools.partial(jax.pmap, static_broadcasted_argnums=(1, 2))
def create_optimizer(model, learning_rate, beta):
  optimizer_def = optim.Momentum(learning_rate=learning_rate, beta=beta)
  optimizer = optimizer_def.create(model)
  return optimizer


def onehot(labels, num_classes=10):
  x = (labels[..., None] == jnp.arange(num_classes)[None])
  return x.astype(jnp.float32)


def cross_entropy_loss(logits, labels):
  return -jnp.mean(jnp.sum(onehot(labels) * logits, axis=-1))


def compute_metrics(logits, labels):
  loss = cross_entropy_loss(logits, labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics


@jax.pmap
def train_step(optimizer, batch):
  """Train for a single step."""
  def loss_fn(model):
    logits = model(batch['image'])
    loss = cross_entropy_loss(logits, batch['label'])
    return loss, logits
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  metrics = compute_metrics(logits, batch['label'])
  return optimizer, metrics


@jax.pmap
def eval_step(model, batch):
  logits = model(batch['image'])
  return compute_metrics(logits, batch['label'])


def train_epoch(optimizer, train_ds, batch_size, epoch, rng):
  """Train for a single epoch."""
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = random.permutation(rng, len(train_ds['image']))
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))
  batch_metrics = []
  for perm in perms:
    batch = {k: v[perm] for k, v in train_ds.items()}
    batch = jax_utils.replicate(batch)
    optimizer, metrics = train_step(optimizer, batch)
    batch_metrics.append(metrics)

  # compute mean of metrics across each batch in epoch.
  batch_metrics_np = jax.device_get(batch_metrics)
  # stack all of the metrics from each devioce into
  # numpy arrays directly on a dict, such that, e.g.
  # `batch_metrics_np['loss']` has shape (jax.device_count(), )
  batch_metrics_np = jax.tree_multimap(lambda *xs: onp.array(xs),
                                       *batch_metrics_np)
  epoch_metrics_np = {
      k: onp.mean(batch_metrics_np[k], axis=0)
      for k in batch_metrics_np
  }
  # `epoch_metrics_np` now contains 1D arrays of length `jax.device_count()`
  logging.info('train epoch: %d, loss: %s, accuracy: %s', epoch,
               epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100)

  return optimizer, epoch_metrics_np


def eval_model(model, test_ds):
  metrics = eval_step(model, test_ds)
  metrics = jax.device_get(metrics)
  summary = metrics
  return summary['loss'], summary['accuracy']


def get_datasets():
  """Load MNIST train and test datasets into memory."""
  ds_builder = tfds.builder('mnist')
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.
  return train_ds, test_ds


def train(train_ds, test_ds):
  """Train MNIST to completion."""
  rng = random.PRNGKey(0)

  batch_size = FLAGS.batch_size
  num_epochs = FLAGS.num_epochs
  model_dir = FLAGS.model_dir

  summary_writer = tensorboard.SummaryWriter(model_dir)

  rng, init_rng = random.split(rng, jax.device_count())
  model = create_model(init_rng)
  optimizer = create_optimizer(model, FLAGS.learning_rate, FLAGS.momentum)

  test_ds = jax_utils.replicate(test_ds)

  for epoch in range(1, num_epochs + 1):
    rng, input_rng = random.split(rng)
    optimizer, train_metrics = train_epoch(
        optimizer, train_ds, batch_size, epoch, input_rng)
    loss, accuracy = eval_model(optimizer.target, test_ds)
    # `loss` and `accuracy` are now 1D arrays of length `jax.device_count()`
    logging.info('eval epoch: %d, loss: %s, accuracy: %s',
                 epoch, loss, accuracy * 100)
    summary_writer.scalar('train_loss', train_metrics['loss'], epoch)
    summary_writer.scalar('train_accuracy', train_metrics['accuracy'], epoch)
    summary_writer.scalar('eval_loss', loss, epoch)
    summary_writer.scalar('eval_accuracy', accuracy, epoch)
  summary_writer.flush()
  return optimizer


def main(_):
  train_ds, test_ds = get_datasets()
  train(train_ds, test_ds)


if __name__ == '__main__':
  app.run(main)
