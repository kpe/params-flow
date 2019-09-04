#!/bin/bash

PEP8_IGNORE=E221,E501,W504,W391

pycodestyle --ignore=${PEP8_IGNORE} --exclude=tests,.venv -r --show-source tests params_flow

coverage run --source=params_flow $(which nosetests) --with-doctest tests/
coverage report --show-missing --fail-under=90
