# About:

[![codecov](https://codecov.io/gh/IsaFoster/MasterThesis/branch/main/graph/badge.svg?token=9CWBWHNZML)](https://app.codecov.io/gh/IsaFoster/MasterThesis/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CodeQL](https://github.com/IsaFoster/MasterThesis/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/IsaFoster/MasterThesis/actions/workflows/github-code-scanning/codeql)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/django)](https://test.pypi.org/project/Robustness/)
[![PyPI](https://badge.fury.io/py/robustness.svg)](https://badge.fury.io/py/adversarial-robustness-toolbox)
[![Build](https://github.com/IsaFoster/MasterThesis/actions/workflows/python-package.yml/badge.svg)](https://github.com/IsaFoster/MasterThesis/actions/workflows/python-package.yml)

# Uploat to testPypi:
https://packaging.python.org/en/latest/tutorials/packaging-projects/
api token i password manager

Publish to pypi test: 
update version number

py -m pip install --upgrade build
py -m build
py -m pip install --upgrade twine
py -m twine upload --repository testpypi dist/*

# TODO:
- cleanup .gitignore
- cleanup testing 
- write tests for calculation
- check python compatibility
- .....?
- profit


# Install dependencies
pip install -e .

# Run tests
from root folder:
pytest -W ignore test_....py

# Force recompile
python setup.py install
python -m compileall .