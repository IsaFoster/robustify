from setuptools import find_packages, setup
setup(
    name='Robustness',
    packages=find_packages(include=['Robustness', 'Noise', 'tests']),
    version='0.2.0',
    description='no description',
    author='Isabel Foster',
    license='MIT',
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=['numpy==1.24.2', 
                      'sklearn==0.0', 
                      'matplotlib==3.6.2', 
                      'pandas==1.5.1', 
                      'plotly==5.11.0', 
                      'tqdm==4.64.1',
                      'chart_studio==1.1.0', 
                      'python-dotenv==0.21.0',
                      'tensorflow==2.11.0',
                      'kaleido==0.2.1'], # list of packages that are absolutely needed, not standard library
    setup_requires=['pytest-runner'], # only installed when required to run tests
    tests_require=['pytest==7.2.2'],
    test_suite='tests',
)
    