#!/usr/bin/env python

#
# created by kpe on 08.04.2019 at 9:33 PM
#


from setuptools import setup

import params_flow


with open("README.rst", "r") as fh:
    long_description = fh.read()


setup(name='params-flow',
      version=params_flow.__version__,
      description='Tensorflow Keras utilities for reducing boilerplate code.',
      url='https://github.com/kpe/params-flow/',
      author='kpe',
      author_email='kpe.git@gmailbox.org',
      license='MIT',
      keywords='tensorflow keras',
      packages=['params_flow'],
      package_data={'params_flow': ['tests/*.py']},
      long_description=long_description,
      long_description_content_type="text/x-rst",
      zip_safe=False,
      install_requires=[
          "py-params >= 0.5.2"
      ],
      python_requires=">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*",
      classifiers=[
          "Development Status :: 5 - Production/Stable",
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python",
          "Programming Language :: Python :: 2",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.4",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: Implementation :: CPython",
          "Programming Language :: Python :: Implementation :: PyPy"])
