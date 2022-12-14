from setuptools import find_packages, setup
setup(
    name='Robustness',
    packages=find_packages(include=['Robustness', 'Setup', 'tests']),
    version='0.1.0',
    description='no description',
    author='Isabel Foster',
    license='MIT',
    install_requires=['numpy==1.23.4', 
                      'sklearn==0.0', 
                      'matplotlib==3.6.2', 
                      'pandas==1.5.1', 
                      'plotly==5.11.0', 
                      'tqdm==4.64.1'], # list of packages that are absolutely needed, not standard library
    setup_requires=['pytest-runner'], # only installed when required to run tests
    tests_require=['pytest==7.1.2'],
    test_suite='tests',
)