from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

from setuptools import find_packages, setup
setup(
    name='robustify',
    packages=find_packages(include=['robustify', 'tests']),
    version='0.0.3',
    readme="README.md",
    description="Robustify:" ,
    long_description=long_description,
    author='Isabel Foster',
    license='MIT',
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=['numpy', 
                      'scikit-learn', 
                      'matplotlib', 
                      'pandas', 
                      'plotly', 
                      'tqdm',
                      'chart_studio', 
                      'python-dotenv',
                      'tensorflow',
                      'kaleido',
                      'ipython',
                      'eli5',
                      'lime',
                      'shap',
                      'torch'], # list of packages that are absolutely needed, not standard library
    setup_requires=['pytest-runner'], # only installed when required to run tests
    tests_require=['pytest==7.3.1'],
    test_suite='tests',
)
    
