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
r"""Ensemble labels or scores to get the ensemble label.

Here's an example of running it:

python tensorflow/examples/speech_commands/ensemble_label.py \
--labels=/tmp/speech_commands_train/conv_labels.txt \
--wav=/tmp/speech_dataset/left/a5d485dc_nohash_0.wav

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
from operator import itemgetter


FLAGS = None


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in open(filename).readlines()]


def load_label_csv(csv_file):
  """Loads the model and labels, and runs the inference to print predictions."""
  labels_list = load_labels(FLAGS.labels)
  num_labels = len(labels_list)
  label2idx = {k.replace('_', ''): v for k, v in zip(labels_list, range(num_labels))}

  with open(csv_file, 'rb') as csvfile:
    reader = csv.reader(csvfile)
    label = {}
    skip_header = True
    has_score = False
    for row in reader:
      if skip_header:
        print(row)
        if len(row) == num_labels + 1:
          has_score = True
          assert row[1:] == labels_list

        skip_header = False
        continue
      if has_score:
        label[row[0]] = [float(v) for v in row[1:]]
      else:
        # print(row)
        assert len(row) == 2
        label[row[0]] = [0.0] * num_labels
        label[row[0]][label2idx[row[1]]] = 1.0
    return label


def ensemble_labels(labels, mode):
  final_label = {}
  num_labels = [len(label) for label in labels]
  if len(num_labels) > 1:
    assert any(num_labels[i] == num_labels[0] for i in range(1, len(num_labels)))

  labels_list = [label.replace('_', '') for label in load_labels(FLAGS.labels)]

  for k in labels[0]:
    scores = [label[k] for label in labels]
    if mode == 'argmax':
      score = [sum(i) for i in zip(*scores)]
      index, _ = max(enumerate(score), key=itemgetter(1))
      final_label[k] = labels_list[index]
    else:
      raise NotImplementedError

  return final_label

def main(argv):
  """Entry point for script, converts flags to arguments."""
  labels = [load_label_csv(f) for f in argv[:-1]]
  label = ensemble_labels(labels, FLAGS.mode)
  with open(argv[-1], 'wb') as f:
    f.write("fname,label\n")
    for (k, v) in label.items():
      f.write("{},{}\n".format(k, v))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--labels', type=str, default='', help='Path to file containing labels.')
  parser.add_argument(
    '--mode',
    type=str,
    default='argmax',
    help='Ensemble mode.')

  FLAGS, unparsed = parser.parse_known_args()
  main(unparsed)
