#!/usr/bin/env python

#
# created by kpe on 08.04.2019 at 9:33 PM
#


from setuptools import setup, find_packages

import params_flow


with open("README.rst", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as reader:
    install_requires = list(map(lambda x: x.strip(), reader.readlines()))


setup(name="params-flow",
      version=params_flow.__version__,
      url="https://github.com/kpe/params-flow/",
      description="Tensorflow Keras utilities for reducing boilerplate code.",
      long_description=long_description,
      long_description_content_type="text/x-rst",
      keywords="tensorflow keras",
      license="MIT",
      author="kpe",
      author_email="kpe.git@gmailbox.org",

      packages=find_packages(exclude=["tests"]),

      package_data={"": ["*.txt", "*.rst"]},
      zip_safe=True,

      install_requires=install_requires,
      python_requires=">=3.4",
      classifiers=[
          "Development Status :: 5 - Production/Stable",
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.4",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: Implementation :: CPython",
          "Programming Language :: Python :: Implementation :: PyPy"])
