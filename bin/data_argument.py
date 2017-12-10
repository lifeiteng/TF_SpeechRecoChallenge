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


def get_key(line):
  wav_file = line.strip()
  feilds = wav_file.split('/')
  return feilds[-2] + '_' + feilds[-1].replace('.wav', '')


class TrimIndexs(object):
  def __init__(self, silence_probs_ark, sample_rate=16000, frame_shift=10, window_size=8, threshold=0.4):
    self.frame_shift = int(sample_rate * frame_shift / 1000.0)
    self.window_size = window_size
    self.threshold = threshold
    self._indexs = {}
    self._load_probs_ark(silence_probs_ark)

  def _load_probs_ark(self, silence_probs_ark):
    with open(silence_probs_ark) as f:
      # sample_02e85b60_nohash_0  [
      # 0.9961087 0.003891301
      # 0.996166 0.003834
      #
      # 0.6095727 0.3904274 ]
      chunk = []
      key = ''
      for line in f:
        line = line.strip()
        if line.find(' ]') > 0:
          assert key not in self._indexs
          assert key != ''
          chunk.append(float(line.split()[0]))
          self._indexs[key] = self._find_start_end(chunk, key=key)
          chunk = []
        elif line.find('  [') > 0:
          key = line.split()[0]
        else:
          chunk.append(float(line.split()[0]))

  def _find_start_end(self, chunk, key=''):
    start = 0
    end = len(chunk)

    for i in range(0, len(chunk) - self.window_size):
      avg = sum(chunk[i:i + self.window_size]) / self.window_size
      if avg < self.threshold:
        start = max(i - int(self.window_size / 2), start)
        break
    for i in range(len(chunk), self.window_size - 1, -1):
      avg = sum(chunk[i - self.window_size:i]) / self.window_size
      if avg < self.threshold:
        end = min(i + int(self.window_size / 2), end)
        break

    if end < start + 10:
      logger.warn(
        '{} start: {} end: {} probs: {}'.format(key, start, end,
                                                ' '.join([':'.join([str(k), str(v)]) for k, v in enumerate(chunk)])))

    assert end > start
    return (start * self.frame_shift, end * self.frame_shift)

  def get_index(self, key):
    assert key in self._indexs
    return self._indexs[key]


def main(silence_probs_ark, wav_list):
  if not os.path.isfile(wav_list):
    raise ValueError("You must set --data_list <wavfile-list>")

  model_settings = prepare_model_settings(FLAGS.sample_rate, FLAGS.clip_duration_ms)
  desired_samples = model_settings['desired_samples']

  trim_indexs = TrimIndexs(silence_probs_ark, FLAGS.sample_rate, frame_shift=10,
                           window_size=FLAGS.window_size, threshold=FLAGS.threshold)

  with open(wav_list) as f:
    for line in f:
      line = line.strip()
      if not line.endswith('.wav'):
        print("Skip line: {}".format(line))
        continue

      logger.info("Process: {}".format('/'.join(line.split('/')[-2:])))
      data, sr = load_wav_file(line)
      # for db in [25, 30, 35, 40, 50]:
      #   data_trim, index = librosa.effects.trim(data / np.max(data), top_db=db, ref=1.0)
      #   wav_file = line.replace('.wav', '_trim{}.wav'.format(db))
      #   write_wav_file(data[index[0]:index[1]], sr, wav_file)
      index = trim_indexs.get_index(get_key(line))
      wav_file = line.replace('.wav', '_trim.wav')
      write_wav_file(data[index[0]:index[1]], sr, wav_file)

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
  parser = argparse.ArgumentParser("Usage: {} silence_probs.ark wav-list".format(__file__))
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
    '--window_size',
    type=int,
    default=4,
    help='Trim: silence probs window size', )
  parser.add_argument(
    '--threshold',
    type=float,
    default=0.5,
    help='Trim: silence threshold', )

  FLAGS, unparsed = parser.parse_known_args()
  if len(unparsed) != 2:
    parser.print_help()
    exit(1)
  main(unparsed[0], unparsed[1])
