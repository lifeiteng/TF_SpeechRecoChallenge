#!/usr/bin/env python
# -*- encoding: utf-8 -*-
__author__ = 'Feiteng'
import sys
import os
import re
from collections import defaultdict

from optparse import OptionParser
from utils import get_logger
import tensorflow as tf

logger = get_logger(__name__)


class Dataset(object):
  """ Dataset for https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data
  """

  def __init__(self, mode):
    self._mode = mode

  @property
  def mode(self):
    """Returns a value in tf.contrib.learn.ModeKeys.
    """
    return self._mode

  @property
  def labels(self):
    return {'yes': 0, 'no': 1, 'up': 2, 'down': 3,
            'left': 4, 'right': 5, 'on': 6, 'off': 7,
            'stop': 8, 'go': 9, 'silence': 10, 'unknown': 11}

  def __call__(self, data_dir):
    assert os.path.isdir(data_dir)
    self._data_dir = data_dir

    audio_label = []
    for item in tf.gfile.Walk(data_dir):
      if not item[1]:
        pattern = re.compile(r'.*/audio/([a-zA-Z]*)')
        label = pattern.findall(item[0])
        if label and (self.mode == tf.contrib.learn.ModeKeys.INFER or label[0] in self.labels):
          assert len(label) == 1
          label = label[0].strip()
        else:
          logger.warn("SKIP: {}".format(item[0]))
          continue

        for audio in item[2]:
          audio_file = os.path.join(item[0], audio)
          audio_label.append((audio_file, label))
      else:
        logger.warn("Skip: {}".format(item))

    kv = defaultdict(lambda: [])
    for item in audio_label:
      kv[item[1]].append(item[0])
    num_items = len(audio_label)
    logger.info("Load {} items.".format(num_items))
    [logger.info("Key: {:10s}, Count: {}, Percent: {:2f}%".format(k, len(kv[k]), len(kv[k]) * 100.0 / num_items)) for k
     in kv]

    return None, None


if __name__ == '__main__':
  cmd_parser = OptionParser(usage="usage: %prog [options] data_dir")

  cmd_parser.parse_args()
  (opts, argv) = cmd_parser.parse_args()

  if len(argv) != 1:
    cmd_parser.print_help()
    exit(-1)

  # Your code here
  dataset = Dataset(tf.contrib.learn.ModeKeys.TRAIN)
  features, labels = dataset(argv[0])
