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
r"""Simple speech recognition to spot a limited number of keywords.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile

from speech import input_data
from speech import models

FLAGS = None


class AudioProcessor(object):
  """Handles loading, partitioning, and preparing audio training data."""

  def __init__(self, data_dir, model_settings, feature_scaling=''):
    self.data_dir = data_dir
    self.prepare_data_index()
    self.prepare_processing_graph(model_settings)
    self.feature_scaling = feature_scaling

  def prepare_data_index(self):
    # Look through all the subfolders to find audio samples
    search_path = os.path.join(self.data_dir, '*', '*.wav')
    self.data_indexs = []
    for wav_path in gfile.Glob(search_path):
      self.data_indexs.append(wav_path)

  def prepare_processing_graph(self, model_settings):
    """Builds a TensorFlow graph to apply the input distortions.

    Creates a graph that loads a WAVE file, decodes it, scales the volume,
    shifts it in time, adds in background noise, calculates a spectrogram, and
    then builds an MFCC fingerprint from that.

    This must be called with an active TensorFlow session running, and it
    creates multiple placeholder inputs, and one output:

      - wav_filename_placeholder_: Filename of the WAV to load.
      - foreground_volume_placeholder_: How loud the main clip should be.
      - time_shift_padding_placeholder_: Where to pad the clip.
      - time_shift_offset_placeholder_: How much to move the clip in time.
      - background_data_placeholder_: PCM sample data for background noise.
      - background_volume_placeholder_: Loudness of mixed-in background.
      - mfcc_: Output 2D fingerprint of processed audio.

    Args:
      model_settings: Information about the current model being trained.
    """
    desired_samples = model_settings['desired_samples']
    self.wav_filename_placeholder_ = tf.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
    wav_decoder = contrib_audio.decode_wav(
      wav_loader, desired_channels=1, desired_samples=desired_samples)
    # Allow the audio sample's volume to be adjusted.
    self.foreground_volume_placeholder_ = tf.placeholder(tf.float32, [])
    scaled_foreground = tf.multiply(wav_decoder.audio,
                                    self.foreground_volume_placeholder_)
    # Shift the sample's start position, and pad any gaps with zeros.
    self.time_shift_padding_placeholder_ = tf.placeholder(tf.int32, [2, 2])
    self.time_shift_offset_placeholder_ = tf.placeholder(tf.int32, [2])
    padded_foreground = tf.pad(
      scaled_foreground,
      self.time_shift_padding_placeholder_,
      mode='CONSTANT')
    sliced_foreground = tf.slice(padded_foreground,
                                 self.time_shift_offset_placeholder_,
                                 [desired_samples, -1])

    mel_bias_ = tf.contrib.signal.linear_to_mel_weight_matrix(num_mel_bins=model_settings['dct_coefficient_count'],
                                                              num_spectrogram_bins=int(2048 / 2 + 1),
                                                              sample_rate=model_settings['sample_rate'],
                                                              lower_edge_hertz=125,
                                                              upper_edge_hertz=float(
                                                                model_settings['sample_rate'] / 2 - 200))
    spectrogram = tf.abs(tf.contrib.signal.stft(tf.transpose(sliced_foreground),
                                                model_settings['window_size_samples'],
                                                model_settings['window_stride_samples'],
                                                fft_length=2048,
                                                window_fn=tf.contrib.signal.hann_window,
                                                pad_end=False))
    S = tf.matmul(tf.reshape(tf.pow(spectrogram, 2), [-1, 1025]), mel_bias_)
    log_mel_spectrograms = tf.log(tf.maximum(S, 1e-7))

    if model_settings['feature_type'] == 'fbank':
      self.mfcc_ = log_mel_spectrograms
    elif model_settings['feature_type'] == 'mfcc':
      # Compute MFCCs from log_mel_spectrograms.
      self.mfcc_ = tf.contrib.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
    else:
      raise ValueError("not supported feature_type: {}".format(model_settings['feature_type']))

  def set_size(self):
    """Calculates the number of samples in the dataset partition.
    Returns:
      Number of samples in the partition.
    """
    return len(self.data_indexs)

  def get_data(self, how_many, offset, model_settings, sess):
    """Gather samples from the data set, applying transformations as needed.
    Returns:
      List of sample data for the transformed samples, and wav files name.
    """
    # Pick one of the partitions to choose samples from.
    candidates = self.data_indexs
    if how_many == -1:
      sample_count = len(candidates)
    else:
      sample_count = max(0, min(how_many, len(candidates) - offset))
    # Data and labels will be populated and returned.
    data = np.zeros((sample_count, model_settings['fingerprint_size']))
    wav_files = []

    # Use the processing graph we created earlier to repeatedly to generate the
    # final output sample data we'll use in training.
    for i in xrange(offset, offset + sample_count):
      # Pick which audio sample to use.
      sample_file = candidates[i]
      time_shift_amount = 0
      if time_shift_amount > 0:
        time_shift_padding = [[time_shift_amount, 0], [0, 0]]
        time_shift_offset = [0, 0]
      else:
        time_shift_padding = [[0, -time_shift_amount], [0, 0]]
        time_shift_offset = [-time_shift_amount, 0]
      input_dict = {self.wav_filename_placeholder_: sample_file,
                    self.time_shift_padding_placeholder_: time_shift_padding,
                    self.time_shift_offset_placeholder_: time_shift_offset}

      input_dict[self.foreground_volume_placeholder_] = 1
      # Run the graph to produce the output audio.
      data[i - offset, :] = sess.run(self.mfcc_, feed_dict=input_dict).flatten()
      wav_files.append(sample_file)

    return input_data.AudioProcessor.apply_feature_scaling(data, self.feature_scaling,
                                                           model_settings['dct_coefficient_count']), wav_files


def main(_):
  # We want to see all the logging messages for this tutorial.
  tf.logging.set_verbosity(tf.logging.INFO)

  sess = tf.InteractiveSession()
  model_settings = models.prepare_model_settings(
    len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
    FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
    FLAGS.window_stride_ms, FLAGS.dct_coefficient_count, FLAGS.feature_type)

  audio_processor = AudioProcessor(FLAGS.data_dir, model_settings)

  fingerprint_size = model_settings['fingerprint_size']
  fingerprint_input = tf.placeholder(tf.float32, [None, fingerprint_size], name='fingerprint_input')

  logits = models.create_model(
    fingerprint_input,
    model_settings,
    FLAGS.model_architecture,
    hparam_string=FLAGS.hparams,
    is_training=False)
  softmax = tf.nn.softmax(logits, name='labels_softmax')

  tf.global_variables_initializer().run()

  checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)
  if checkpoint_path:
    models.load_variables_from_checkpoint(sess, checkpoint_path)
  else:
    tf.logging.fatal("Not find checkpoint.")

  set_size = audio_processor.set_size()
  tf.logging.info('set_size=%d', set_size)

  with gfile.GFile(FLAGS.output_csv, 'w') as wf:
    wf.write("fname,{}\n".format(','.join(input_data.prepare_words_list(FLAGS.wanted_words.split(',')))))
    for i in xrange(0, set_size, FLAGS.batch_size):
      test_fingerprints, test_wavfiles = audio_processor.get_data(
        FLAGS.batch_size, i, model_settings, sess)
      probs = sess.run(softmax,
                       feed_dict={
                         fingerprint_input: test_fingerprints,
                       })
      for k, wav_file in enumerate(test_wavfiles):
        wf.write("%s,%s\n" % (wav_file.split('/')[-1], ','.join([str(v) for v in probs[k]])))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--data_dir',
    type=str,
    default='/tmp/speech_dataset/',
    help="""\
      Where to download the speech training data to.
      """)
  parser.add_argument(
    '--time_shift_ms',
    type=float,
    default=100.0,
    help="""\
      Range to randomly shift the training audio by in time.
      """)
  parser.add_argument(
    '--sample_rate',
    type=int,
    default=16000,
    help='Expected sample rate of the wavs', )
  parser.add_argument(
    '--clip_duration_ms',
    type=int,
    default=1000,
    help='Expected duration in milliseconds of the wavs', )
  parser.add_argument(
    '--window_size_ms',
    type=float,
    default=30.0,
    help='How long each spectrogram timeslice is', )
  parser.add_argument(
    '--window_stride_ms',
    type=float,
    default=10.0,
    help='How long each spectrogram timeslice is', )
  parser.add_argument(
    '--dct_coefficient_count',
    type=int,
    default=40,
    help='How many bins to use for the MFCC fingerprint', )
  parser.add_argument(
    '--batch_size',
    type=int,
    default=128,
    help='How many items to train with at once', )
  parser.add_argument(
    '--wanted_words',
    type=str,
    default='yes,no,up,down,left,right,on,off,stop,go',
    help='Words to use (others will be added to an unknown label)', )
  parser.add_argument(
    '--train_dir',
    type=str,
    default='/tmp/speech_commands_train',
    help='Directory to write event logs and checkpoint.')
  parser.add_argument(
    '--model_architecture',
    type=str,
    default='conv',
    help='What model architecture to use')
  parser.add_argument(
    '--hparams',
    type=str,
    default='',
    help='Hyper parameters string')
  parser.add_argument(
    '--output_csv',
    type=str,
    default='',
    help='Output file name')
  parser.add_argument(
    '--feature_scaling',
    type=str,
    default='',  # '' 'cmvn'
    help='Feature normalization')
  parser.add_argument(
    '--feature_type',
    type=str,
    default='mfcc',  #
    help='Feature type (e.g. mfcc or fbank)')

  FLAGS, unparsed = parser.parse_known_args()
  if not FLAGS.output_csv:
    raise ValueError("must set --output_csv")

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
