from setuptools import find_packages, setup
setup(
    name='Robustness',
    packages=find_packages(include=['Robustness']),
    version='0.1.0',
    description='My first Python library',
    author='Isabel Foster',
    license='MIT',
    install_requires=['numpy', 'sklearn', 'matplotlib'], # list of packages that are absolutely needed, not standard library
    setup_requires=['pytest-runner'], # only installed when required to run tests
    tests_require=['pytest==7.1.2'],
    test_suite='tests',
)