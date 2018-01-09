# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Contains definitions for the preactivation form of Residual Networks.

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs, is_training, data_format):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  inputs = tf.layers.batch_normalization(
    inputs=inputs, axis=1 if data_format == 'channels_first' else -1,
    momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
    scale=True, training=is_training, fused=True)
  inputs = tf.nn.relu(inputs)
  return inputs


def batch_norm(inputs, is_training, data_format, name=None):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  inputs = tf.layers.batch_normalization(
    inputs=inputs, axis=1 if data_format == 'channels_first' else -1,
    momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
    scale=True, training=is_training, fused=True, name=name)
  return inputs


def relu_batch_norm(inputs, is_training, data_format):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  inputs = tf.nn.relu(inputs)
  inputs = tf.layers.batch_normalization(
    inputs=inputs, axis=1 if data_format == 'channels_first' else -1,
    momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
    scale=True, training=is_training, fused=True)
  return inputs


def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  if isinstance(kernel_size, int):
    pad_total = kernel_size - 1
    pad_beg1 = pad_total // 2
    pad_end1 = pad_total - pad_beg1
    pad_beg2 = pad_beg1
    pad_end2 = pad_end1
  else:
    assert isinstance(kernel_size, tuple)
    pad_beg1 = kernel_size[0] // 2
    pad_end1 = kernel_size[0] - pad_beg1
    pad_beg2 = kernel_size[1] // 2
    pad_end2 = kernel_size[1] - pad_beg2

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg1, pad_end1], [pad_beg2, pad_end2]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg1, pad_end1],
                                    [pad_beg2, pad_end2], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides,
                         data_format, dilation_rate=(1, 1),
                         regularizer=None, use_bias=False):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
    inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
    padding=('SAME' if strides == 1 else 'VALID'), use_bias=use_bias,
    kernel_initializer=tf.variance_scaling_initializer(),
    kernel_regularizer=regularizer,
    data_format=data_format, dilation_rate=dilation_rate)


def fixed_padding_1d(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [0, 0]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [0, 0], [0, 0]])
  return padded_inputs


def conv1d_fixed_padding(inputs, filters, kernel_size, strides,
                         data_format, dilation_rate=(1, 1),
                         regularizer=None, use_bias=False):
  return tf.layers.conv1d(
    inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
    padding='SAME', use_bias=use_bias,
    kernel_initializer=tf.variance_scaling_initializer(),
    kernel_regularizer=regularizer,
    data_format=data_format, dilation_rate=(dilation_rate[0]))


def create_hparams(hparam_string=None):
  """Create model hyperparameters. Parse nondefault from given string."""
  hparams = tf.contrib.training.HParams(
    # The name of the architecture to use.
    add_batch_norm=False,
    pooling_type="average",
    kernel_size=3,
    add_dropout=False,
    add_first_batch_norm=False,
    freeze_first_batch_norm=False,
    freeze_batch_norm=False,
    use_conv1d=False,
    use_dilation_conv=True,

    #########################
    # Resnet Hyperparameters#
    #########################
    resnet_blocks=6,  # Number of resnet blocks
    resnet_filters=45,  # Number of filters per conv in resnet blocks
    # paper: Identity Mappings in Deep Residual Networks
    # resnet type(Figure 4): ['pre-activation'(e), 'ReLU before addition'(c)]
    resnet_type='c',
    bottleneck_sizes=[0],  # if not [0], add bottleneck layer
    regularizer_l1_scale=0.0,
    regularizer_l2_scale=0.0
  )

  if hparam_string:
    tf.logging.info('Parsing command line hparams: %s', hparam_string)
    hparams.parse(hparam_string)

  tf.logging.info('Final parsed hparams: %s', hparams.values())
  if hparams.resnet_type not in ['c', 'e']:
    raise ValueError("not supported resnet_type: {} (not in ['c', 'e'])".format(hparams.resnet_type))
  if not hparams.bottleneck_sizes:
    assert any(s >= 0 for s in hparams.bottleneck_sizes)
  assert isinstance(hparams.bottleneck_sizes, list)

  return hparams


def resnet_generator(num_classes, dropout_prob=1.0,
                     data_format="channels_last", hparam_string=''):
  """Generator for ResNet15 DEEP RESIDUAL LEARNING FOR SMALL-FOOTPRINT KEYWORD SPOTTING.

  Args:
    num_classes: The number of possible classes for image classification.
    data_format: The input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.

  Returns:
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the ResNet model.

  Raises:
    ValueError: If `resnet_size` is invalid.
  """
  hparams = create_hparams(hparam_string)
  dropout_prob = 1 - dropout_prob

  regularizer = None
  if hparams.regularizer_l2_scale > 0:
    regularizer = tf.contrib.layers.l2_regularizer(scale=hparams.regularizer_l2_scale)
  elif hparams.regularizer_l1_scale > 0:
    regularizer = tf.contrib.layers.l1_regularizer(scale=hparams.regularizer_l1_scale)

  def model(inputs, is_training):
    """Constructs the ResNet model given the inputs."""
    _, input_time_size, input_frequency_size, _ = inputs.get_shape().as_list()
    conv_fn = conv2d_fixed_padding
    if hparams.use_conv1d:
      conv_fn = conv1d_fixed_padding
      # Rank 4 -> 3
      inputs = tf.squeeze(inputs, axis=-1, name='inputs_squeezed')

    freeze_batch_norm = is_training and (not hparams.freeze_batch_norm)

    tf.summary.histogram('inputs', inputs)
    if hparams.add_first_batch_norm:
      with tf.variable_scope("initial_norm"):
        inputs = batch_norm(inputs, is_training=freeze_batch_norm and (not hparams.freeze_first_batch_norm),
                            data_format=data_format, name='initial_norm')
      tf.summary.histogram('inputs_batchnorm', inputs)
    inputs = tf.layers.dropout(inputs, rate=dropout_prob, training=is_training)

    with tf.variable_scope('initial_conv'):
      inputs = conv_fn(
        inputs=inputs, filters=hparams.resnet_filters, kernel_size=(3, 10), strides=(1, 4),
        data_format=data_format, regularizer=regularizer)
      inputs = tf.identity(inputs, 'initial_conv')
    inputs = tf.layers.dropout(inputs, rate=dropout_prob, training=is_training)

    def _residual_block(inputs, filters=hparams.resnet_filters, dilations=(0, 0), name=None):
      with tf.variable_scope(name):
        shortcut = inputs
        with tf.variable_scope("conv1"):
          if hparams.resnet_type == 'e':
            if hparams.add_batch_norm:
              inputs = batch_norm_relu(inputs, is_training=freeze_batch_norm, data_format=data_format)
            else:
              inputs = tf.nn.relu(inputs)
          inputs = conv_fn(
            inputs=inputs, filters=filters, kernel_size=3, strides=1,
            data_format=data_format, dilation_rate=(dilations[0], dilations[0]),
            regularizer=regularizer)
          if hparams.resnet_type == 'c':
            if hparams.add_batch_norm:
              inputs = batch_norm_relu(inputs, is_training=freeze_batch_norm, data_format=data_format)
            else:
              inputs = tf.nn.relu(inputs)

        with tf.variable_scope("conv2"):
          if hparams.resnet_type == 'e':
            if hparams.add_batch_norm:
              inputs = batch_norm_relu(inputs, is_training=freeze_batch_norm, data_format=data_format)
            else:
              inputs = tf.nn.relu(inputs)
          inputs = conv_fn(
            inputs=inputs, filters=filters, kernel_size=3, strides=1,
            data_format=data_format, dilation_rate=(dilations[1], dilations[1]),
            regularizer=regularizer)
          if hparams.resnet_type == 'c':
            if hparams.add_batch_norm:
              inputs = batch_norm_relu(inputs, is_training=freeze_batch_norm, data_format=data_format)
            else:
              inputs = tf.nn.relu(inputs)

        return tf.identity(inputs + shortcut)

    for x in range(1, hparams.resnet_blocks + 1):
      def _dilation(i):
        if hparams.use_dilation_conv:
          return int(math.pow(2, math.floor(i / 3)))
        return 1

      inputs = _residual_block(inputs=inputs, filters=hparams.resnet_filters,
                               dilations=(_dilation(2 * x), _dilation(2 * x + 1)),
                               name='block_layer{}'.format(x))
      inputs = tf.layers.dropout(inputs, rate=dropout_prob, training=is_training)

    with tf.variable_scope("final_conv"):
      inputs = conv_fn(inputs, filters=hparams.resnet_filters, kernel_size=3, strides=1,
                       data_format=data_format, regularizer=regularizer)
      if hparams.add_batch_norm:
        inputs = batch_norm(inputs, is_training=freeze_batch_norm, data_format=data_format)

    if hparams.use_conv1d:
      pooling_fn = tf.layers.average_pooling1d
      pool_size = 4
      if hparams.pooling_type == 'max':
        pooling_fn = tf.layers.max_pooling1d
      inputs = pooling_fn(
        inputs=inputs, pool_size=pool_size, strides=(2), padding='VALID', data_format=data_format)

      assert (input_time_size - pool_size) % 2 == 0
      output_size = int((input_time_size - pool_size) / 2) + 1
      inputs = tf.reshape(inputs, [-1, hparams.resnet_filters * output_size])
      inputs = tf.identity(inputs, 'final_avg_pool')
    else:
      inputs_max = tf.reduce_max(inputs, [1, 2], name='GlobalMaximumPooling')
      inputs_avg = tf.reduce_mean(inputs, [1, 2], name='GlobalAveragePooling')
      inputs = tf.concat([inputs_max, inputs_avg], axis=1)
      inputs = tf.identity(inputs, 'final_pool')
    if hparams.bottleneck_sizes[0] > 0:
      for i, size in enumerate(hparams.bottleneck_sizes):
        inputs = tf.layers.dense(inputs=inputs, units=size, name="final_bn{}".format(i))

    inputs = tf.layers.dropout(inputs, rate=dropout_prob, training=is_training)
    inputs = tf.layers.dense(inputs=inputs, units=num_classes)
    inputs = tf.identity(inputs, 'final_logits')
    return inputs

  return model


def create_densenet_hparams(hparam_string=None):
  """Create model hyperparameters. Parse nondefault from given string."""
  hparams = tf.contrib.training.HParams(
    add_first_batch_norm=True,
    freeze_first_batch_norm=False,
    freeze_batch_norm=False,
    inital_filters=16,
    dense_blocks=3,  # Number of dense blocks
    num_layers=40,  # 40, 100
    growth_rate=12,  # 12, 24, 40
    add_bottleneck_layer=False,
    theta=1,
    regularizer_l2_scale=0.0,
    regularizer_l1_scale=0.0
  )

  if hparam_string:
    tf.logging.info('Parsing command line hparams: %s', hparam_string)
    hparams.parse(hparam_string)

  return hparams


def densenet_generator(num_classes, dropout_prob=1.0,
                       data_format="channels_last", hparam_string=''):
  """Generator for DenseNet: Densely Connected Convolutional Networks.
  # https://github.com/liuzhuang13/DenseNet/blob/631bff19eecbc0aa75d25cb4b7324f3fc66439cb/models/densenet.lua

  Args:
    resnet_size: A single integer for the size of the ResNet model.
    num_classes: The number of possible classes for image classification.
    data_format: The input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.

  Returns:
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the ResNet model.

  Raises:
    ValueError: If `resnet_size` is invalid.
  """
  hparams = create_densenet_hparams(hparam_string)
  dropout_prob = 1 - dropout_prob

  regularizer = None
  if hparams.regularizer_l2_scale > 0:
    regularizer = tf.contrib.layers.l2_regularizer(scale=hparams.regularizer_l2_scale)
  elif hparams.regularizer_l1_scale > 0:
    regularizer = tf.contrib.layers.l1_regularizer(scale=hparams.regularizer_l1_scale)

  def model(inputs, is_training):
    """Constructs the ResNet model given the inputs."""
    _, input_time_size, input_frequency_size, _ = inputs.get_shape().as_list()
    is_training = is_training and (not hparams.freeze_batch_norm)
    tf.summary.histogram('inputs', inputs)
    if hparams.add_first_batch_norm:
      with tf.variable_scope("InitialNorm"):
        inputs = batch_norm(inputs, is_training=is_training and (not hparams.freeze_first_batch_norm),
                            data_format=data_format, name='initial_norm')
      tf.summary.histogram('inputs_batchnorm', inputs)
    inputs = tf.layers.dropout(inputs, rate=dropout_prob, training=is_training)

    with tf.variable_scope('InitialConv'):
      inputs = conv2d_fixed_padding(
        inputs=inputs, filters=hparams.inital_filters, kernel_size=(3, 10), strides=(1, 4),
        data_format=data_format, regularizer=regularizer)
      inputs = tf.identity(inputs, 'initial_conv')
    inputs = tf.layers.dropout(inputs, rate=dropout_prob, training=is_training)

    def _dense_block(inputs, name=None):
      with tf.variable_scope(name):
        preceding_inputs = [inputs]
        for i in range(hparams.num_layers):
          with tf.variable_scope('Layers{}'.format(i)):
            inputs = tf.concat(preceding_inputs, axis=-1)
            if hparams.add_bottleneck_layer:
              # DenseNet-B
              inputs = batch_norm_relu(inputs, is_training=is_training, data_format=data_format)
              inputs = conv2d_fixed_padding(inputs=inputs, filters=4 * hparams.growth_rate,
                                            kernel_size=1, strides=1, data_format=data_format,
                                            regularizer=regularizer)

            inputs = batch_norm_relu(inputs, is_training=is_training, data_format=data_format)
            inputs = conv2d_fixed_padding(
              inputs=inputs, filters=hparams.growth_rate, kernel_size=3, strides=1,
              data_format=data_format)
            inputs = tf.layers.dropout(inputs, rate=dropout_prob, training=is_training)

            preceding_inputs.append(inputs)
        return inputs

    for x in range(1, hparams.dense_blocks + 1):
      inputs = _dense_block(inputs=inputs, name='DenseBlock{}'.format(x))
      inputs = batch_norm_relu(inputs, is_training=is_training, data_format=data_format)

      if x != hparams.dense_blocks:
        with tf.variable_scope('TransitionLayer{}'.format(x)):
          # 1x1 conv
          num_filters = hparams.growth_rate
          if hparams.theta < 1:
            # DenseNet-C
            num_filters = int(num_filters * hparams.theta)
          inputs = conv2d_fixed_padding(
            inputs=inputs, filters=num_filters, kernel_size=1, strides=1,
            data_format=data_format, regularizer=regularizer)
          inputs = tf.layers.dropout(inputs, rate=dropout_prob, training=is_training)
          # pooling
          inputs = tf.layers.average_pooling2d(inputs, pool_size=2, padding='same',
                                               strides=2, data_format=data_format)

    # global average pooling
    assert data_format == 'channels_last'
    inputs_max = tf.reduce_max(inputs, [1, 2], name='GlobalMaximumPooling')
    inputs_avg = tf.reduce_mean(inputs, [1, 2], name='GlobalAveragePooling')
    inputs = tf.concat([inputs_max, inputs_avg], axis=1)
    inputs = tf.identity(inputs, 'FinalPool')

    inputs = tf.layers.dropout(inputs, rate=dropout_prob, training=is_training)
    inputs = tf.layers.dense(inputs=inputs, units=num_classes)
    inputs = tf.identity(inputs, 'FinalLogits')
    return inputs

  return model


def create_resnetft_hparams(hparam_string=None):
  """Create model hyperparameters. Parse nondefault from given string."""
  hparams = tf.contrib.training.HParams(
    # The name of the architecture to use.
    add_batch_norm=False,
    pooling_type="average",
    kernel_size=3,
    add_dropout=False,
    add_first_batch_norm=False,
    freeze_first_batch_norm=False,
    freeze_batch_norm=False,
    use_conv1d=False,
    use_dilation_conv=True,

    #########################
    # Resnet Hyperparameters#
    #########################
    resnet_blocks=6,  # Number of resnet blocks
    resnet_filters=45,  # Number of filters per conv in resnet blocks
    # paper: Identity Mappings in Deep Residual Networks
    # resnet type(Figure 4): ['pre-activation'(e), 'ReLU before addition'(c)]
    resnet_type='c',
    bottleneck_sizes=[0],  # if not [0], add bottleneck layer
    regularizer_l1_scale=0.0,
    regularizer_l2_scale=0.0,
    use_bias=False,
  )

  if hparam_string:
    tf.logging.info('Parsing command line hparams: %s', hparam_string)
    hparams.parse(hparam_string)

  tf.logging.info('Final parsed hparams: %s', hparams.values())
  if hparams.resnet_type not in ['c', 'e']:
    raise ValueError("not supported resnet_type: {} (not in ['c', 'e'])".format(hparams.resnet_type))
  if not hparams.bottleneck_sizes:
    assert any(s >= 0 for s in hparams.bottleneck_sizes)
  assert isinstance(hparams.bottleneck_sizes, list)

  return hparams


def resnetft_generator(num_classes, dropout_prob=1.0,
                       data_format="channels_last", hparam_string=''):
  """Generator for ResNetFT.

  Args:
    num_classes: The number of possible classes for image classification.
    data_format: The input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.

  Returns:
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the ResNet model.

  Raises:
    ValueError: If `resnet_size` is invalid.
  """
  hparams = create_resnetft_hparams(hparam_string)
  dropout_prob = 1 - dropout_prob

  regularizer = None
  if hparams.regularizer_l2_scale > 0:
    regularizer = tf.contrib.layers.l2_regularizer(scale=hparams.regularizer_l2_scale)
  elif hparams.regularizer_l1_scale > 0:
    regularizer = tf.contrib.layers.l1_regularizer(scale=hparams.regularizer_l1_scale)

  def model(inputs, is_training):
    """Constructs the ResNet model given the inputs."""
    _, input_time_size, input_frequency_size, _ = inputs.get_shape().as_list()
    conv_fn = conv1d_fixed_padding
    inputs = tf.squeeze(inputs, axis=-1, name='inputs_squeezed')

    freeze_batch_norm = is_training and (not hparams.freeze_batch_norm)

    tf.summary.histogram('inputs', inputs)
    if hparams.add_first_batch_norm:
      with tf.variable_scope("initial_norm"):
        inputs = batch_norm(inputs, is_training=freeze_batch_norm and (not hparams.freeze_first_batch_norm),
                            data_format=data_format, name='initial_norm')
      tf.summary.histogram('inputs_batchnorm', inputs)
    inputs = tf.layers.dropout(inputs, rate=dropout_prob, training=is_training)

    with tf.variable_scope('initial_conv'):
      inputs = conv_fn(
        inputs=inputs, filters=hparams.resnet_filters, kernel_size=4, strides=1,
        data_format=data_format, regularizer=regularizer, use_bias=hparams.use_bias)
      inputs = tf.identity(inputs, 'initial_conv')
    inputs = tf.layers.dropout(inputs, rate=dropout_prob, training=is_training)

    def _residual_block(inputs, filters=hparams.resnet_filters, dilations=(0, 0), name=None):
      with tf.variable_scope(name):
        shortcut = inputs
        with tf.variable_scope("conv1"):
          if hparams.resnet_type == 'e':
            if hparams.add_batch_norm:
              inputs = batch_norm_relu(inputs, is_training=freeze_batch_norm, data_format=data_format)
            else:
              inputs = tf.nn.relu(inputs)
          inputs = conv_fn(
            inputs=inputs, filters=filters, kernel_size=4, strides=1,
            data_format=data_format, dilation_rate=(dilations[0], dilations[0]),
            regularizer=regularizer, use_bias=hparams.use_bias)
          if hparams.resnet_type == 'c':
            if hparams.add_batch_norm:
              inputs = batch_norm_relu(inputs, is_training=freeze_batch_norm, data_format=data_format)
            else:
              inputs = tf.nn.relu(inputs)

        with tf.variable_scope("conv2"):
          if hparams.resnet_type == 'e':
            if hparams.add_batch_norm:
              inputs = batch_norm_relu(inputs, is_training=freeze_batch_norm, data_format=data_format)
            else:
              inputs = tf.nn.relu(inputs)
          inputs = conv_fn(
            inputs=inputs, filters=filters, kernel_size=4, strides=1,
            data_format=data_format, dilation_rate=(dilations[1], dilations[1]),
            regularizer=regularizer, use_bias=hparams.use_bias)
          if hparams.resnet_type == 'c':
            if hparams.add_batch_norm:
              inputs = batch_norm_relu(inputs, is_training=freeze_batch_norm, data_format=data_format)
            else:
              inputs = tf.nn.relu(inputs)

        return tf.identity(inputs + shortcut)

    for x in range(1, hparams.resnet_blocks + 1):
      def _dilation(i):
        if hparams.use_dilation_conv:
          return int(math.pow(2, math.floor(i / 3)))
        return 1

      inputs = _residual_block(inputs=inputs, filters=hparams.resnet_filters,
                               dilations=(_dilation(2 * x), _dilation(2 * x + 1)),
                               name='block_layer{}'.format(x))
      inputs = tf.layers.dropout(inputs, rate=dropout_prob, training=is_training)

    with tf.variable_scope("final_conv"):
      inputs = conv_fn(inputs, filters=hparams.resnet_filters, kernel_size=4, strides=1,
                       data_format=data_format, regularizer=regularizer, use_bias=hparams.use_bias)
      if hparams.add_batch_norm:
        inputs = batch_norm(inputs, is_training=freeze_batch_norm, data_format=data_format)

    inputs_max = tf.reduce_max(inputs, [1], name='GlobalMaximumPooling')
    inputs_avg = tf.reduce_mean(inputs, [1], name='GlobalAveragePooling')
    inputs = tf.concat([inputs_max, inputs_avg], axis=1)
    inputs = tf.identity(inputs, 'final_pool')
    if hparams.bottleneck_sizes[0] > 0:
      for i, size in enumerate(hparams.bottleneck_sizes):
        inputs = tf.layers.dense(inputs=inputs, units=size, name="final_bn{}".format(i))

    inputs = tf.layers.dropout(inputs, rate=dropout_prob, training=is_training)
    inputs = tf.layers.dense(inputs=inputs, units=num_classes)
    inputs = tf.identity(inputs, 'final_logits')
    return inputs

  return model
