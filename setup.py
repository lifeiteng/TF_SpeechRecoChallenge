"""
Python package setup file.
"""

from setuptools import setup

setup(
  name="TF_Speech",
  version="0.2.0",
  extras_require={'tensorflow': ['tensorflow'],
                  'tensorflow with gpu': ['tensorflow-gpu']},
)
