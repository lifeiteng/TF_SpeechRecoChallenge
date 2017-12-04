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

This is a self-contained example script that will train a very basic audio
recognition model in TensorFlow. It downloads the necessary training data and
runs with reasonable defaults to train within a few hours even only using a CPU.
For more information, please see
https://www.tensorflow.org/tutorials/audio_recognition.

It is intended as an introduction to using neural networks for audio
recognition, and is not a full speech recognition system. For more advanced
speech systems, I recommend looking into Kaldi. This network uses a keyword
detection style to spot discrete words from a small vocabulary, consisting of
"yes", "no", "up", "down", "left", "right", "on", "off", "stop", and "go".

To run the training process, use:

bazel run tensorflow/examples/speech_commands:train

This will write out checkpoints to /tmp/speech_commands_train/, and will
download over 1GB of open source training data, so you'll need enough free space
and a good internet connection. The default data is a collection of thousands of
one-second .wav files, each containing one spoken word. This data set is
collected from https://aiyprojects.withgoogle.com/open_speech_recording, please
consider contributing to help improve this and other models!

As training progresses, it will print out its accuracy metrics, which should
rise above 90% by the end. Once it's complete, you can run the freeze script to
get a binary GraphDef that you can easily deploy on mobile applications.

If you want to train on your own data, you'll need to create .wavs with your
recordings, all at a consistent length, and then arrange them into subfolders
organized by label. For example, here's a possible file structure:

my_wavs >
  up >
    audio_0.wav
    audio_1.wav
  down >
    audio_2.wav
    audio_3.wav
  other>
    audio_4.wav
    audio_5.wav

You'll also need to tell the script what labels to look for, using the
`--wanted_words` argument. In this case, 'up,down' might be what you want, and
the audio in the 'other' folder would be used to train an 'unknown' category.

To pull this all together, you'd run:

bazel run tensorflow/examples/speech_commands:train -- \
--data_dir=my_wavs --wanted_words=up,down

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import input_data
import models
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.python.platform import gfile

FLAGS = None


def main(_):
  # We want to see all the logging messages for this tutorial.
  tf.logging.set_verbosity(tf.logging.INFO)

  # Start a new TensorFlow session.
  sess = tf.InteractiveSession()

  # Begin by making sure we have the training data we need. If you already have
  # training data of your own, use `--data_url= ` on the command line to avoid
  # downloading.
  model_settings = models.prepare_model_settings(
    len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
    FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
    FLAGS.window_stride_ms, FLAGS.dct_coefficient_count, FLAGS.resnet_size)

  audio_processor = input_data.AudioProcessor(
    FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
    FLAGS.unknown_percentage,
    FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
    FLAGS.testing_percentage, model_settings)

  fingerprint_size = model_settings['fingerprint_size']
  fingerprint_input = tf.placeholder(tf.float32, [None, fingerprint_size], name='fingerprint_input')

  logits, dropout_prob = models.create_model(
    fingerprint_input,
    model_settings,
    FLAGS.model_architecture,
    is_training=False)
  softmax = tf.nn.softmax(logits, name='labels_softmax')

  tf.global_variables_initializer().run()

  checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)
  if checkpoint_path:
    models.load_variables_from_checkpoint(sess, checkpoint_path)
  else:
    tf.logging.fatal("Not find checkpoint.")

  set_size = audio_processor.set_size('testing')
  tf.logging.info('set_size=%d', set_size)

  with gfile.GFile(FLAGS.output, 'w') as wf:
    wf.write("fname,{}\n".format(','.join(input_data.prepare_words_list(FLAGS.wanted_words.split(',')))))
    for i in xrange(0, set_size, FLAGS.batch_size):
      test_fingerprints, test_wavfiles = audio_processor.get_data(
        FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
      probs = sess.run(softmax,
                       feed_dict={
                         fingerprint_input: test_fingerprints,
                         dropout_prob: 1.0
                       })
      for i, wav_file in enumerate(test_wavfiles):
        wf.write("%s,%s\n" % (wav_file.split('/')[-1], ','.join([str(v) for v in probs[i]])))


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
    '--background_volume',
    type=float,
    default=0.1,
    help="""\
      How loud the background noise should be, between 0 and 1.
      """)
  parser.add_argument(
    '--background_frequency',
    type=float,
    default=0.8,
    help="""\
      How many of the training samples have background noise mixed in.
      """)
  parser.add_argument(
    '--silence_percentage',
    type=float,
    default=10.0,
    help="""\
      How much of the training data should be silence.
      """)
  parser.add_argument(
    '--unknown_percentage',
    type=float,
    default=10.0,
    help="""\
      How much of the training data should be unknown words.
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
    '--resnet_size',
    type=int,
    default=32,
    help='Residual model layers.')
  parser.add_argument(
    '--model_architecture',
    type=str,
    default='conv',
    help='What model architecture to use')
  parser.add_argument(
    '--output',
    type=str,
    default='',
    help='Output file name')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
