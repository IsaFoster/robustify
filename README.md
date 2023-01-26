# About:

[![codecov](https://codecov.io/gh/IsaFoster/MasterThesis/branch/main/graph/badge.svg?token=9CWBWHNZML)](https://app.codecov.io/gh/IsaFoster/MasterThesis/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![CodeQL](https://github.com/isafoster/MasterThesis/workflows/CodeQL/badge.svg)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/django)](https://test.pypi.org/project/Robustness/)
[![PyPI](https://badge.fury.io/py/robustness.svg)](https://badge.fury.io/py/adversarial-robustness-toolbox)

![Continuous Integration](https://github.com/IsaFoster/MasterThesis/workflows/Continuous%20Integration/badge.svg)


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