from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

from setuptools import find_packages, setup
setup(
    name='RobustifyToolkit',
    packages=find_packages(include=['robustify', 'robustify.noise', 'robustify.utils', 'tests']),
    version='0.1.0',
    readme="README.md",
    description="Robustify:" ,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Isabel Foster',
    license='MIT',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
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
                      'torch']
    setup_requires=['pytest-runner'], 
    tests_require=['pytest==7.3.1'],
    test_suite='tests',
)
    