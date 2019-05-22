#!/bin/bash

PEP8_IGNORE=E221,E501,W504,W391

pep8 --ignore=$PEP8_IGNORE --exclude=tests,.venv -r --show-source .

coverage run --source=params_flow /home/kpe/proj/local/params-flow.kpe/.venv/bin/nosetests --with-doctest tests/
coverage report --show-missing --fail-under=100
