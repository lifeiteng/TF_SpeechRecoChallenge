#!/usr/bin/env python
# Copyright 2017 Feiteng

r""" Data argument
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os

import librosa
import numpy as np


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


def prepare_model_settings(sample_rate, clip_duration_ms):
  """Calculates common settings needed for all models.

  Args:
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.

  Returns:
    Dictionary containing common settings.
  """
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  return {
    'desired_samples': desired_samples,
  }


def load_wav_file(wav_file):
  return librosa.load(wav_file, sr=None)


def write_wav_file(data, sr, wav_file):
  np.clip(data, -1.0, 1.0, out=data)
  librosa.output.write_wav(wav_file, data.astype(np.float32), sr)


def pad_and_write(data, sr, wav_file, desired_samples=16000):
  pad_len = desired_samples - data.shape[0]
  begin = np.random.randint(abs(pad_len))
  if pad_len >= 0:
    end = pad_len - begin
    data = np.pad(data, [begin, end], mode='constant')
  elif (pad_len / desired_samples) < 0.02:
    data = data[begin:begin + desired_samples]
  else:
    logger.warn("Not store data: len = {} desired_samples = {}".format(data.shape[0], desired_samples))
    return

  assert data.shape[0] == desired_samples
  write_wav_file(data, sr, wav_file)


def main(wav_list):
  if not os.path.isfile(wav_list):
    raise ValueError("You must set --data_list <wavfile-list>")

  model_settings = prepare_model_settings(FLAGS.sample_rate, FLAGS.clip_duration_ms)
  desired_samples = model_settings['desired_samples']

  with open(wav_list) as f:
    for line in f:
      line = line.strip()
      if not line.endswith('.wav'):
        print("Skip line: {}".format(line))
        continue

      logger.info("Process: {}".format('/'.join(line.split('/')[-2:])))
      data, sr = load_wav_file(line)
      for db in [25, 30, 35, 40, 50]:
        data_trim, index = librosa.effects.trim(data / np.max(data), top_db=db, ref=1.0)
        wav_file = line.replace('.wav', '_trim{}.wav'.format(db))
        write_wav_file(data[index[0]:index[1]], sr, wav_file)
      # data_trim, index = librosa.effects.trim(data, top_db=60)
      # # Random PAD -> store
      # wav_file = line.replace('.wav', '_trim.wav')
      # write_wav_file(data_trim, sr, wav_file)

      # # speed
      # for sp in [1.1, 1.2]:
      #   data_sp = librosa.effects.time_stretch(data, sp)
      #   wav_file = line.replace('.wav', '_sp{}.wav'.format(sp))
      #   pad_and_write(data_sp, sr, wav_file, desired_samples=desired_samples)

      # for sp in [0.8, 0.9]:
      #   data_sp = librosa.effects.time_stretch(data_trim, sp)
      #   # data_sp length -> PAD -> Store
      #   wav_file = line.replace('.wav', '_sp{}.wav'.format(sp))
      #   pad_and_write(data_sp, sr, wav_file, desired_samples=desired_samples)

      # volume
      for v in [0.8, 0.9, 1.1, 1.2]:
        pass


if __name__ == '__main__':
  parser = argparse.ArgumentParser("Usage: {} wav-list".format(__file__))
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

  FLAGS, unparsed = parser.parse_known_args()
  if len(unparsed) != 1:
    parser.print_help()
    exit(1)
  main(unparsed[0])
