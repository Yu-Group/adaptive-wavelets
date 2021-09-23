#!/usr/bin/env python

from os import path

import setuptools

path_to_repo = path.abspath(path.dirname(__file__))
with open(path.join(path_to_repo, 'readme.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(name='awave',
                 version='0.1',
                 description='Adaptive wavelets',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 author='Wooseok Ha, Chandan Singh',
                 author_email='haywse@berkeley.edu',
                 url='https://github.com/Yu-Group/adaptive-wavelet-distillation',
                 packages=setuptools.find_packages(exclude=['tests']),
                 install_requires=[
                     'torch>=1.0',
                     'numpy',
                     'pywavelets'
                 ],
                 python_requires='>=3.6',
                 classifiers=[
                     "Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License",
                     "Operating System :: OS Independent",
                 ],
                 )
