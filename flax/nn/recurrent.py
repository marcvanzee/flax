# Lint as: python3
"""Library with additional Flax recurrent neural network components."""

import abc

from . import activation
from . import base
from . import initializers
from . import linear

from jax import random

from typing import Any, Optional, Sequence, Text, Tuple, Type

import jax
import jax.numpy as jnp

import numpy as np


class RNNCellBase(base.Module):
  """RNN cell base class."""

  @staticmethod
  @abc.abstractmethod
  def initialize_carry(rng, batch_dims, size, init_fn=initializers.zeros):
    """initialize the RNN cell carry.
    Args:
      rng: random number generator passed to the init_fn.
      batch_dims: a tuple providing the shape of the batch dimensions.
      size: the size or number of features of the memory.
      init_fn: initializer function for the carry.
    Returns:
      An initialized carry for the given RNN cell.
    """
    pass


class LSTMCell(RNNCellBase):
  """LSTM cell."""

  def apply(self, carry, inputs,
            gate_fn=activation.sigmoid, activation_fn=activation.tanh,
            kernel_init=linear.default_kernel_init,
            recurrent_kernel_init=initializers.orthogonal(),
            bias_init=initializers.zeros):
    r"""A long short-term memory (LSTM) cell.
    the mathematical definition of the cell is as follows
    .. math::
        \begin{array}{ll}
        i = \sigma(W_{ii} x + W_{hi} h + b_{hi}) \\
        f = \sigma(W_{if} x + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + W_{hg} h + b_{hg}) \\
        o = \sigma(W_{io} x + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * \tanh(c') \\
        \end{array}
    where x is the input, h is the output of the previous time step, and c is
    the memory.
    Args:
      carry: the hidden state of the LSTM cell,
        initialized using `LSTMCell.initialize_carry`.
      inputs: an ndarray with the input for the current time step.
        All dimensions except the final are considered batch dimensions.
      gate_fn: activation function used for gates (default: sigmoid)
      activation_fn: activation function used for output and memory update
        (default: tanh).
      kernel_init: initializer function for the kernels that transform
        the input (default: lecun_normal).
      recurrent_kernel_init: initializer function for the kernels that transform
        the hidden state (default: orthogonal).
      bias_init: initializer for the bias parameters (default: zeros)
    Returns:
      A tuple with the new carry and the output.
    """
    c, h = carry
    hidden_features = h.shape[-1]
    # input and recurrent layers are summed so only one needs a bias.
    dense_h = linear.Dense.partial(
        inputs=h, features=hidden_features, bias=True,
        kernel_init=recurrent_kernel_init, bias_init=bias_init)
    dense_i = linear.Dense.partial(
        inputs=inputs, features=hidden_features, bias=False,
        kernel_init=kernel_init)
    i = gate_fn(dense_i(name='ii') + dense_h(name='hi'))
    f = gate_fn(dense_i(name='if') + dense_h(name='hf'))
    g = activation_fn(dense_i(name='ig') + dense_h(name='hg'))
    o = gate_fn(dense_i(name='io') + dense_h(name='ho'))
    new_c = f * c + i * g
    new_h = o * activation_fn(new_c)
    return (new_c, new_h), new_h

  @staticmethod
  def initialize_carry(rng, batch_dims, size, init_fn=initializers.zeros):
    """initialize the RNN cell carry.
    Args:
      rng: random number generator passed to the init_fn.
      batch_dims: a tuple providing the shape of the batch dimensions.
      size: the size or number of features of the memory.
      init_fn: initializer function for the carry.
    Returns:
      An initialized carry for the given RNN cell.
    """
    key1, key2 = random.split(rng)
    mem_shape = batch_dims + (size,)
    return init_fn(key1, mem_shape), init_fn(key2, mem_shape)


class GRUCell(RNNCellBase):
  """GRU cell."""

  def apply(self, carry, inputs,
            gate_fn=activation.sigmoid, activation_fn=activation.tanh,
            kernel_init=linear.default_kernel_init,
            recurrent_kernel_init=initializers.orthogonal(),
            bias_init=initializers.zeros):
    r"""Gated recurrent unit (GRU) cell.
    the mathematical definition of the cell is as follows
    .. math::
        \begin{array}{ll}
        r = \sigma(W_{ir} x + W_{hr} h + b_{hr}) \\
        z = \sigma(W_{iz} x + W_{hz} h + b_{hz}) \\
        n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h' = (1 - z) * n + z * h
        \end{array}
    where x is the input and h, is the output of the previous time step.
    Args:
      carry: the hidden state of the LSTM cell,
        initialized using `GRUCell.initialize_carry`.
      inputs: an ndarray with the input for the current time step.
        All dimensions except the final are considered batch dimensions.
      gate_fn: activation function used for gates (default: sigmoid)
      activation_fn: activation function used for output and memory update
        (default: tanh).
      kernel_init: initializer function for the kernels that transform
        the input (default: lecun_normal).
      recurrent_kernel_init: initializer function for the kernels that transform
        the hidden state (default: orthogonal).
      bias_init: initializer for the bias parameters (default: zeros)
    Returns:
      A tuple with the new carry and the output.
    """
    h = carry
    hidden_features = h.shape[-1]
    # input and recurrent layers are summed so only one needs a bias.
    dense_h = linear.Dense.partial(
        inputs=h, features=hidden_features, bias=False,
        kernel_init=recurrent_kernel_init, bias_init=bias_init)
    dense_i = linear.Dense.partial(
        inputs=inputs, features=hidden_features, bias=True,
        kernel_init=kernel_init, bias_init=bias_init)
    r = gate_fn(dense_i(name='ir') + dense_h(name='hr'))
    z = gate_fn(dense_i(name='iz') + dense_h(name='hz'))
    # add bias because the linear transformations aren't directly summed.
    n = activation_fn(dense_i(name='in') + r * dense_h(name='hn', bias=True))
    new_h = (1. - z) * n + z * h
    return new_h, new_h

  @staticmethod
  def initialize_carry(rng, batch_dims, size, init_fn=initializers.zeros):
    """initialize the RNN cell carry.
    Args:
      rng: random number generator passed to the init_fn.
      batch_dims: a tuple providing the shape of the batch dimensions.
      size: the size or number of features of the memory.
      init_fn: initializer function for the carry.
    Returns:
      An initialized carry for the given RNN cell.
    """
    mem_shape = batch_dims + (size,)
    return init_fn(rng, mem_shape)


def _cell_name(cell_def: Any, layer_num: int, suffix: Text = '') -> Text:
  """Helper function to name recurrent cells."""
  cell_name = cell_def._default_name() or cell_def.__name__  # pylint:disable=protected-access
  return f'{cell_name}_layer{layer_num}{suffix}'


def sequence_mask(lengths: jnp.ndarray, max_length: int) -> jnp.ndarray:
  """Computes a boolean mask over sequence positions for each given length.

  Example:
  ```
  sequence_mask([1, 2], 3)
  [[True, False, False],
   [True, True, False]]
  ```

  Args:
    lengths: The length of each sequence. <int>[batch_size]
    max_length: The width of the boolean mask. Must be >= max(lengths).

  Returns:
    A mask with shape: <bool>[lengths.size, max_length] indicating which
    positions are valid for each sequence.
  """
  return jnp.arange(max_length) < jnp.expand_dims(lengths, 1)


@jax.vmap
def flip_sequences(inputs: jnp.ndarray, lengths: jnp.ndarray) -> jnp.array:
  """Flips a sequence of inputs along the time dimension."""
  idxs = (jnp.arange(inputs.shape[0] - 1, -1, -1) + lengths) % inputs.shape[0]
  return inputs[idxs]


def unroll_cell(
    cell: Any,
    inputs: jnp.ndarray,
    lengths: jnp.ndarray,
    initial_state: Any,
    recurrent_dropout_mask: Optional[jnp.ndarray] = None,
    initializing: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Unrolls a recurrent cell."""

  def _step_fn(carry, step_inputs):
    cell_state, step_index = carry
    if recurrent_dropout_mask is not None:
      # TODO(flaxnlp): make this work for GRU cells.
      assert cell.__name__ == 'LSTMCell', \
          'Recurrent dropout only available for LSTM cells.'
      output = cell_state[1] * recurrent_dropout_mask
      cell_state = (cell_state[0], output)
    new_cell_state, output = cell(cell_state, step_inputs)

    # Pass the new state, unless we're done processing based on length.
    def select_carried_state(old_state, new_state):
      return jnp.where(step_index < jnp.expand_dims(lengths, 1), new_state,
                       old_state)

    # LSTM state is a tuple (c, h). Select c and h separately and recombine.
    # TODO(flaxnlp): make this work for GRU cells.
    carried_cell_state = tuple(
        select_carried_state(*s) for s in zip(cell_state, new_cell_state))

    return (carried_cell_state, step_index + 1), output

  init_carry = (initial_state, jnp.array(0))
  if initializing:
    # Initialize parameters before scan.
    _step_fn(init_carry, inputs[:, 0])

  (final_state, _), outputs = flax.jax_utils.scan_in_dim(
      _step_fn, init=init_carry, xs=inputs, axis=1)
  return outputs, final_state


def unroll_cell_backwards(
    cell: Any,
    inputs: jnp.ndarray,
    lengths: jnp.ndarray,
    initial_state: Any,
    recurrent_dropout_mask: Optional[jnp.ndarray] = None,
    initializing: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Flips the inputs, unrolls the cell, and unflips the outputs."""
  outputs, final_state = unroll_cell(
      cell, flip_sequences(inputs, lengths), lengths, initial_state,
      recurrent_dropout_mask, initializing=initializing)
  outputs = flip_sequences(outputs, lengths)  # Back to original input order.
  return outputs, final_state


class RNNBase(base.Module):
  """Base RNN."""

  def apply(self,
            cell_def: Type[RNNCellBase],
            inputs: jnp.ndarray,
            lengths: jnp.ndarray,
            initial_states: Optional[Sequence[Any]] = None,
            hidden_size: int = None,
            num_layers: int = 1,
            dropout_rate: float = 0.,
            recurrent_dropout_rate: float = 0.,
            bidirectional: bool = False,
            train: bool = False) -> Tuple[jnp.ndarray, Sequence[Any]]:
    """Processes the input sequence using a given recurrent cell.

    Args:
      cell_def: A recurrent cell such as LSTMCell or GRUCell.
      inputs: The input sequence <float32>[batch_size, sequence_length, ...]
      lengths: The lengths of each sequence in the batch. <int64>[batch_size]
      initial_states: The initial states for the cells. You must provide
        `num_layers` initial states (when using bidirectional, `num_layers *
        2`).
        These must be ordered in the following way: (layer_0_forward,
          layer_0_backward, layer_1_forward, layer_1_backward, ...). If None,
          all initial states will be initialized with zeros.
      hidden_size: The size of each recurrent cell.
      num_layers: The number of stacked recurrent layers. The output of the
        first layer, with optional dropout applied, feeds into the next layer.
      dropout_rate: Dropout rate to be applied between LSTM layers. Only applies
        when num_layers > 1.
      recurrent_dropout_rate: Dropout rate to be applied on the hidden state at
        each time step repeating the same dropout mask.
      bidirectional: Process the sequence left-to-right and right-to-left and
        concatenate the outputs from the two directions.
      train: Enables dropout between layers.

    Returns:
      The sequence of all outputs for the final layer, and a list of final
      states for each cell and direction, ordered first by layer number and then
      by direction (first forward, then backward, if bidirectional). For some
      cells like LSTMCell a state consists of an (c, h) tuple, while for others
      cells it only contains a single vector (h,).
    """
    batch_size = inputs.shape[0]
    final_states = []
    num_cells = num_layers * (2 if bidirectional else 1)
    if initial_states is None:  # Initialize with zeros.
      rng = jax.random.PRNGKey(0)
      initial_states = [
          cell_def.initialize_carry(rng, (batch_size,), hidden_size)
          for _ in range(num_cells)
      ]
    assert len(initial_states) == num_layers * (2 if bidirectional else 1), \
        'Please provide `num_layers` (*2 if bidirectional) initial states.'

    cell_idx = 0
    for layer_idx in range(num_layers):
      recurrent_dropout_mask = (None, None)
      if train and recurrent_dropout_rate > 0.:
        rng = nn.make_rng()
        directions = 2 if bidirectional else 1
        recurrent_dropout_mask = random.bernoulli(
            rng, p=1 - recurrent_dropout_rate,
            shape=(directions, batch_size, hidden_size))

        # Scale mask.
        recurrent_dropout_mask = recurrent_dropout_mask / (
            1.0 - recurrent_dropout_rate)

      cell = cell_def.shared(name=_cell_name(cell_def, layer_idx))
      outputs, final_state = unroll_cell(
          cell, inputs, lengths, initial_states[cell_idx],
          recurrent_dropout_mask[0], initializing=self.is_initializing())
      final_states.append(final_state)
      cell_idx += 1

      if bidirectional:
        cell = cell_def.shared(name=_cell_name(cell_def, layer_idx, '_rev'))
        backward_outputs, backward_final_state = unroll_cell_backwards(
            cell, inputs, lengths, initial_states[cell_idx],
            recurrent_dropout_mask[1], initializing=self.is_initializing())
        outputs = jnp.concatenate([outputs, backward_outputs], axis=-1)
        final_states.append(backward_final_state)
        cell_idx += 1

      inputs = outputs  # For the next layer, current outputs become the inputs.
      if train and dropout_rate > 0.:
        inputs = nn.dropout(inputs, rate=dropout_rate)

    return outputs, final_states


class LSTM(RNNBase):
  """LSTM."""

  def apply(self,
            inputs: jnp.ndarray,
            lengths: jnp.ndarray,
            initial_states: Optional[Sequence[Any]] = None,
            hidden_size: int = None,
            num_layers: int = 1,
            dropout_rate: float = 0.,
            recurrent_dropout_rate: float = 0.,
            bidirectional: bool = False,
            train: bool = False,
            cell_module: base.Module = LSTMCell,
            **cell_kwargs) -> Tuple[jnp.ndarray, Sequence[Any]]:
    """Processes an input sequence with an LSTM cell.

    Example usage::

      inputs = np.random.normal(size=(2, 3, 4))
      lengths = np.array([1, 3])
      outputs, final_states = LSTM(inputs, lengths, hidden_size=10)


    Args:
      inputs: The input sequence <float32>[batch_size, sequence_length, ...]
      lengths: The lengths of each sequence in the batch. <int64>[batch_size]
      initial_states: The initial states for the cells. You must provide
        `num_layers` initial states (when using bidirectional, `num_layers *
        2`). These must be ordered in the following way: (layer_0_forward,
          layer_0_backward, layer_1_forward, layer_1_backward, ...). If None,
          all initial states will be initialized with zeros.
      hidden_size: The size of each recurrent cell.
      num_layers: The number of stacked recurrent layers. The output of the
        first layer, with optional dropout applied, feeds into the next layer.
      dropout_rate: Dropout rate to be applied between LSTM layers. Only applies
        when num_layers > 1.
      recurrent_dropout_rate: Dropout rate to be applied on the hidden state at
        each time step repeating the same dropout mask.
      bidirectional: Process the sequence left-to-right and right-to-left and
        concatenate the outputs from the two directions.
      train: Enables dropout between layers.
      cell_module: The LSTM cell to use.
      **cell_kwargs: Keyword arguments passed to the LSTM cell. This can be used
        to specify custom initializers.

    Returns:
      The sequence of all outputs for the final layer, and a list of final
      states (h, c) for each cell and direction, ordered first by layer number
      and then by direction (first forward, then backward, if bidirectional).
    """
    cell = cell_module.partial(**cell_kwargs)
    return super().apply(
        cell,
        inputs,
        lengths,
        initial_states,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        recurrent_dropout_rate=recurrent_dropout_rate,
        bidirectional=bidirectional,
        train=train,
    )


class OptimizedLSTMCell(RNNCellBase):
  """More efficient LSTM Cell that concatenates state components before matmul.

  Parameters are compatible with `flax.nn.LSTMCell`.
  """

  class DummyDense(base.Module):
    """Dummy module for creating parameters matching `flax.nn.Dense`."""

    def apply(self,
              inputs,
              features,
              kernel_init,
              bias_init,
              bias=True):
      k = self.param('kernel', (inputs.shape[-1], features), kernel_init)
      b = (self.param('bias', (features,), bias_init)
           if bias else jnp.zeros((features,)))
      return k, b

  def apply(self,
            carry,
            inputs,
            gate_fn=activation.sigmoid,
            activation_fn=activation.tanh,
            kernel_init=linear.default_kernel_init,
            recurrent_kernel_init=initializers.orthogonal(),
            bias_init=initializers.zeros):
    r"""A long short-term memory (LSTM) cell.

    the mathematical definition of the cell is as follows
    .. math::
        \begin{array}{ll}
        i = \sigma(W_{ii} x + W_{hi} h + b_{hi}) \\
        f = \sigma(W_{if} x + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + W_{hg} h + b_{hg}) \\
        o = \sigma(W_{io} x + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * \tanh(c') \\
        \end{array}
    where x is the input, h is the output of the previous time step, and c is
    the memory.

    Args:
      carry: the hidden state of the LSTM cell, initialized using
        `LSTMCell.initialize_carry`.
      inputs: an ndarray with the input for the current time step. All
        dimensions except the final are considered batch dimensions.
      gate_fn: activation function used for gates (default: sigmoid)
      activation_fn: activation function used for output and memory update
        (default: tanh).
      kernel_init: initializer function for the kernels that transform
        the input (default: lecun_normal).
      recurrent_kernel_init: initializer function for the kernels that transform
        the hidden state (default: orthogonal).
      bias_init: initializer for the bias parameters (default: zeros)

    Returns:
      A tuple with the new carry and the output.
    """
    c, h = carry
    hidden_features = h.shape[-1]

    def _concat_dense(inputs, params, use_bias=True):
      kernels, biases = zip(*params.values())
      kernel = jnp.asarray(jnp.concatenate(kernels, axis=-1), jnp.float32)

      y = jax.lax.dot_general(
          inputs, kernel,
          (((inputs.ndim - 1,), (0,)), ((), ())))
      if use_bias:
        bias = jnp.asarray(jnp.concatenate(biases, axis=-1), jnp.float32)
        y = y + bias
      ys = jnp.split(y, np.cumsum([b.shape[0] for b in biases[:-1]]), axis=-1)
      return dict(zip(params.keys(), ys))

    # Create the params in the same order as flax.nn.LSTMCell for initialization
    # compatibility.
    dense_params_h = {}
    dense_params_i = {}
    for component in ['i', 'f', 'g', 'o']:
      dense_params_i[component] = OptimizedLSTMCell.DummyDense(
          inputs=inputs, features=hidden_features, bias=False,
          kernel_init=kernel_init, bias_init=bias_init,
          name=f'i{component}')
      dense_params_h[component] = OptimizedLSTMCell.DummyDense(
          inputs=h, features=hidden_features, bias=True,
          kernel_init=recurrent_kernel_init, bias_init=bias_init,
          name=f'h{component}')
    dense_h = _concat_dense(h, dense_params_h, use_bias=True)
    dense_i = _concat_dense(inputs, dense_params_i, use_bias=False)

    i = gate_fn(dense_h['i'] + dense_i['i'])
    f = gate_fn(dense_h['f'] + dense_i['f'])
    g = activation_fn(dense_h['g'] + dense_i['g'])
    o = gate_fn(dense_h['o'] + dense_i['o'])

    new_c = f * c + i * g
    new_h = o * activation_fn(new_c)
    return (new_c, new_h), new_h

  @staticmethod
  def initialize_carry(rng, batch_dims, size, init_fn=initializers.zeros):
    """initialize the RNN cell carry.

    Args:
      rng: random number generator passed to the init_fn.
      batch_dims: a tuple providing the shape of the batch dimensions.
      size: the size or number of features of the memory.
      init_fn: initializer function for the carry.

    Returns:
      An initialized carry for the given RNN cell.
    """
    key1, key2 = random.split(rng)
    mem_shape = batch_dims + (size,)
    return init_fn(key1, mem_shape), init_fn(key2, mem_shape)