#!/usr/bin/env python
# -*- encoding: utf-8 -*-
__author__ = 'Feiteng'
import sys
import os
from optparse import OptionParser
from utils import get_logger

logger = get_logger(__name__)


class Dataset(object):
  """ Dataset for https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data
  """
  def __init__(self, data_dir):
    assert os.path.isdir(data_dir)
    self._data_dir = data_dir

  def __call__(self, *args, **kwargs):
    raise NotImplementedError


class TrainEvalDataset(Dataset):
  def __init__(self, data_dir):
    super(TrainEvalDataset, self).__init__(data_dir)

  def __call__(self, *args, **kwargs):
    pass


class InferDataset(Dataset):
  def __init__(self, data_dir):
    super(InferDataset, self).__init__(data_dir)

  def __call__(self, *args, **kwargs):
    pass


if __name__ == '__main__':
  cmd_parser = OptionParser(usage="usage: %prog [options] recognized_file reference_file")
  cmd_parser.add_option('-V', '--verbose',
                        action="store", type="int", dest="V", default=0, help='Verbose level')

  cmd_parser.parse_args()
  (opts, argv) = cmd_parser.parse_args()

  if len(argv) != 2:
    cmd_parser.print_help()
    exit(-1)

  # Your code here
