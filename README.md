# About:

<!---Codecov public repo:
[![codecov](https://codecov.io/gh/IsaFoster/MasterThesis/branch/main/graph/badge.svg)](https://app.codecov.io/gh/IsaFoster/MasterThesis/)
Codecov private repo:--->
<a href="https://codecov.io/gh/IsaFoster/MasterThesis" > 
 <img src="https://codecov.io/gh/IsaFoster/MasterThesis/branch/main/graph/badge.svg?token=9CWBWHNZML"/> 
</a>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Må tilpasses:
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/django)](https://test.pypi.org/project/Robustness/)
[![PyPI](https://badge.fury.io/py/robustness.svg)](https://badge.fury.io/py/adversarial-robustness-toolbox)
![Continuous Integration](https://github.com/Trusted-AI/adversarial-robustness-toolbox/workflows/Continuous%20Integration/badge.svg)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/5090/badge)](https://bestpractices.coreinfrastructure.org/projects/5090)


IDK: 
![CodeQL](https://github.com/Trusted-AI/adversarial-robustness-toolbox/workflows/CodeQL/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/adversarial-robustness-toolbox/badge/?version=latest)](http://adversarial-robustness-toolbox.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


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