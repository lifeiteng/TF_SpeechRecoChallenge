#!/usr/bin/env python
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
import random
from collections import defaultdict
from operator import itemgetter

from mlens.ensemble import BlendEnsemble

from speech import infer
from speech import input_data

FLAGS = None


class color:
  d = {}
  RESET_SEQ = ""

  def __init__(self, c):
    if c == True:
      self.d['K'] = "\033[0;30m"  # black
      self.d['R'] = "\033[0;31m"  # red
      self.d['G'] = "\033[0;32m"  # green
      self.d['Y'] = "\033[0;33m"  # yellow
      self.d['B'] = "\033[0;34m"  # blue
      self.d['M'] = "\033[0;35m"  # magenta
      self.d['C'] = "\033[0;36m"  # cyan
      self.d['W'] = "\033[0;37m"  # white
      self.RESET_SEQ = "\033[0m"
    else:
      self.d['K'] = "["
      self.d['R'] = "<"
      self.d['G'] = "["
      self.d['Y'] = "["
      self.d['B'] = "["
      self.d['M'] = "["
      self.d['C'] = "["
      self.d['W'] = "["
      self.RESET_SEQ = "]"

  def c_string(self, color, string):
    return self.d[color] + string + self.RESET_SEQ


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in open(filename).readlines()]


def get_basename(wav_name):
  # audio/clip_59c59305c.wav
  # clip_59c59305c_vp1.2.wav
  class_name, wav_name = infer.deprefix_dirname(wav_name)
  return class_name, '_'.join(wav_name.replace('.wav', '').split('_')[:2]) + '.wav'


def load_score_csv(csv_file):
  """Loads the model and labels, and runs the inference to print predictions."""

  wanted_words = [w for w in input_data.prepare_words_list(FLAGS.wanted_words.split(','))]
  word2idx = {word: i for i, word in enumerate(wanted_words)}

  with open(csv_file, 'rb') as csvfile:
    reader = csv.reader(csvfile)
    label = defaultdict(lambda: {'scores': [], 'class': ''})
    skip_header = True
    has_score = False
    head_words = None
    for row in reader:
      if skip_header:
        print(row)
        head_words = row[1:]
        skip_header = False
        if len(row[1:]) > 2:
          has_score = True
        continue
      if row[0] in label:
        print("repeated wav: {}".format(row[0]))
      class_name, basename = get_basename(row[0])
      if has_score:
        if head_words != wanted_words:
          # merge other words's score to unknown_score
          re_scores = [0.0] * len(wanted_words)
          for i, score in enumerate(row[1:]):
            if head_words[i] not in wanted_words:
              re_scores[word2idx[input_data.UNKNOWN_WORD_LABEL]] += float(score)
            else:
              re_scores[word2idx[head_words[i]]] += float(score)
          label[basename] = {'scores': re_scores, 'class': class_name}
        else:
          label[basename] = {'scores': [float(v) for v in row[1:]], 'class': class_name}
      else:
        assert False
        # # print(row)
        # assert len(row) == 2
        # label[basename].append([0.0] * num_labels)
        # label[basename][-1][label2idx[row[1]]] = 1.0
    return label


colors = color(True)


def get_estimators_list(proba=True):
  return [RandomForestClassifier(random_state=0), SVC(probability=proba)]


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def build_ensemble(proba=True, **kwargs):
  """Return an ensemble."""

  ensemble = BlendEnsemble(**kwargs)
  ensemble.add(get_estimators_list(proba=True), proba=proba)  # Specify 'proba' here
  ensemble.add_meta(LogisticRegression())

  return ensemble


def ensemble_labels(data_train, data_test):
  wanted_words = [w.replace('_', '') for w in input_data.prepare_words_list(FLAGS.wanted_words.split(','))]
  word2idx = {word: i for i, word in enumerate(wanted_words)}

  UNKNOWN_WORD_LABEL = input_data.UNKNOWN_WORD_LABEL.replace('_', '')
  final_label = {}
  num_labels = [len(label) for label in data_test]
  if len(num_labels) > 1:
    assert any(num_labels[i] == num_labels[0] for i in range(1, len(num_labels)))

  X = []
  y = []

  X_train = []
  y_train = []
  for i, k in enumerate(data_test[0].keys()):
    scores = []
    for score in data_test:
      scores.append(score[k]['scores'])
    if i % 10000 == 0:
      print("Processed {} utters".format(i))

    num_scores = len(scores)
    score = [sum([pow(s, FLAGS.factor) for s in ss]) / num_scores for ss in zip(*scores)]

    index, value = max(enumerate(score), key=itemgetter(1))

    final_label[k] = wanted_words[index]
    if wanted_words[index] not in wanted_words:
      print("WARN: {} -> {}".format(wanted_words[index], UNKNOWN_WORD_LABEL))
      final_label[k] = UNKNOWN_WORD_LABEL

    X.append(score)
    y.append(word2idx[final_label[k]])

    if value > 0.95 and (final_label[k] == 'silence'):
      X_train.append(score)
      y_train.append(word2idx[final_label[k]])

  print("INFO: got {} silence data for train.".format(len(y_train)))

  for k in data_train:
    X_train.append(data_train[k]["scores"])
    class_name = data_train[k]["class"]
    if class_name in wanted_words:
      y_train.append(word2idx[class_name])
    else:
      y_train.append(word2idx[UNKNOWN_WORD_LABEL])

  X_y_train = list(zip(X_train, y_train))

  random.shuffle(X_y_train)

  X_train, y_train = zip(*X_y_train)

  ensemble = build_ensemble(proba=True)
  ensemble.fit(X_train, y_train)
  print("Accuracy Train:\n%r" % accuracy_score(ensemble.predict(X_train), y_train))

  preds = ensemble.predict(X)
  print("Accuracy  Test:\n%r" % accuracy_score(preds, y))

  return final_label


def main(argv):
  """Entry point for script, converts flags to arguments."""
  data_train = load_score_csv(argv[0])

  data_test = [load_score_csv(f) for f in argv[1:-1]]
  label = ensemble_labels(data_train, data_test)
  with open(argv[-1], 'wb') as f:
    f.write("fname,label\n")
    for (k, v) in label.items():
      dirname, wavfile = infer.deprefix_dirname(k)
      f.write("{},{}\n".format(wavfile, v))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--wanted_words',
    type=str,
    default='yes,no,up,down,left,right,on,off,stop,go',
    help='Words to use (others will be added to an unknown label)')
  parser.add_argument(
    '--debug',
    action='store_true',
    default=False,
    help='Output score instead of label')
  parser.add_argument(
    '--tune',
    action='store_true',
    default=False,
    help='Tune label')
  parser.add_argument(
    '--factor',
    type=float,
    default=0.5,
    help='Ensemble factor value.')
  parser.add_argument(
    '--output_data_dir',
    type=str,
    default='',
    help='Select training data from test dataset.')
  parser.add_argument(
    '--threshold',
    type=float,
    default=0.9,
    help='Threshold for select training data from test dataset')

  FLAGS, unparsed = parser.parse_known_args()
  if len(unparsed) < 3:
    parser.print_usage()
    raise ValueError("at least two inputs.")

  main(unparsed)
