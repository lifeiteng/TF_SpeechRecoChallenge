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
import os
import random
import time
from collections import defaultdict
from operator import itemgetter
import logging

from mlens.ensemble import BlendEnsemble

from speech import infer
from speech import input_data

FLAGS = None


def get_logger(name, time=True):
  logger = logging.getLogger(name)
  logger.setLevel(logging.INFO)
  handler = logging.StreamHandler()
  handler.setLevel(logging.INFO)
  formatter = logging.Formatter("{}[%(filename)s:%(lineno)s - "
                                "%(funcName)s - %(levelname)s ] %(message)s".format('%(asctime)s ' if time else ''))
  handler.setFormatter(formatter)
  logger.addHandler(handler)
  return logger


logger = get_logger(__name__)


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
      class_name, _ = get_basename(row[0])
      if has_score:
        if head_words != wanted_words:
          # merge other words's score to unknown_score
          re_scores = [0.0] * len(wanted_words)
          for i, score in enumerate(row[1:]):
            if head_words[i] not in wanted_words:
              re_scores[word2idx[input_data.UNKNOWN_WORD_LABEL]] += float(score)
            else:
              re_scores[word2idx[head_words[i]]] += float(score)
          label[row[0]] = {'scores': re_scores, 'class': class_name}
        else:
          label[row[0]] = {'scores': [float(v) for v in row[1:]], 'class': class_name}
      else:
        assert False
        # # print(row)
        # assert len(row) == 2
        # label[basename].append([0.0] * num_labels)
        # label[basename][-1][label2idx[row[1]]] = 1.0
    return label


colors = color(True)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def get_estimators_list(proba=True):
  return [RandomForestClassifier(random_state=0), LogisticRegression()]


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
  y_value = []

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
    y_value.append(value)

    if final_label[k] == 'silence':
      X_train.append(score)
      y_train.append(word2idx[final_label[k]])

  print("INFO: got {} silence data for train.".format(len(y_train)))

  print("INFO: data_train {}".format(len(data_train)))
  for k in data_train:
    X_train.append(data_train[k]["scores"])
    class_name = data_train[k]["class"]
    if class_name in wanted_words:
      y_train.append(word2idx[class_name])
    else:
      y_train.append(word2idx[UNKNOWN_WORD_LABEL])

  # balance samples
  for v in set(y_train):
    print("INFO: label: {:8s} count: {}".format(wanted_words[v], y_train.count(v)))

  mini_count = min([y_train.count(v) for v in set(y_train)])

  X_y_train = list(zip(X_train, y_train))
  random.shuffle(X_y_train)
  X_train, y_train = zip(*X_y_train)

  ensemble = build_ensemble(proba=True)
  ensemble.fit(X_train, y_train)
  print("Accuracy Train:\n%r" % accuracy_score(ensemble.predict(X_train), y_train))

  preds = ensemble.predict(X)
  acc = accuracy_score(preds, y)
  print("Accuracy  Test:\n%r" % acc)

  num_changed = 0
  if acc > 0.9:
    # for (ens, pred, k, score, value) in zip(preds, y, data_test[0].keys(), X, y_value):
    #   if value < 0.7:
    #     final_label[k] = wanted_words[int(ens)]

    # if FLAGS.debug:
    UNKNOWN_WORD_INDEX = word2idx[UNKNOWN_WORD_LABEL]
    mac_data_dir = '/Users/feiteng/Geek/Kaggle/TF_Speech/test/audio/'
    for (ens, pred, k, score, value) in zip(preds, y, data_test[0].keys(), X, y_value):
      # final_label[k] = wanted_words[int(ens)]
      ens = int(ens)
      if ens != pred and value < 0.7 and (score[pred] < 2 * score[UNKNOWN_WORD_INDEX]):
        # unknown tuning
        if final_label[k] in [UNKNOWN_WORD_LABEL]:
          continue
        final_label[k] = wanted_words[int(ens)]
        num_changed += 1

        if FLAGS.debug:
          # _, wav_name = infer.deprefix_dirname(k)
          wav_name = k
          os.system("play -V0 {}{}".format(mac_data_dir, wav_name))
          print(
            "INFO: {} Origin: {} Ensemble: {}".format(
              ' '.join(["%8s: %.4f\n" % (k, v) for k, v in zip(wanted_words, score)]),
              colors.c_string('G', wanted_words[pred]),
              colors.c_string('R', wanted_words[ens])))
          time.sleep(4)
    print("INFO: {} / {} = {:4f} label changed.".format(num_changed, len(final_label),
                                                        num_changed * 1.0 / len(final_label)))

  return final_label


def main(argv):
  """Entry point for script, converts flags to arguments."""
  data_train = load_score_csv(argv[0])

  data_test = [load_score_csv(f) for f in argv[1:-1]]
  label = ensemble_labels(data_train, data_test)
  with open(argv[-1], 'wb') as f:
    f.write("fname,label\n")
    for (k, v) in label.items():
      # class_name, basename = get_basename(k)
      basename = k
      f.write("{},{}\n".format(basename, v))


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
