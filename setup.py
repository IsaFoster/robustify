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
    install_requires=['numpy<=1.24', 
                      'scikit-learn==1.2.2', 
                      'matplotlib==3.7.1', 
                      'pandas==2.0.1', 
                      'plotly==5.14.1', 
                      'tqdm==4.64.1',
                      'chart_studio==1.1.0', 
                      'python-dotenv==1.0.0',
                      'tensorflow==2.11.1',
                      'kaleido==0.2.1',
                      'ipython==8.11.0',
                      'eli5==0.13.0',
                      'lime==0.2.0.1',
                      'shap==0.41.0',
                      'torch==2.0.1'],
    setup_requires=['pytest-runner'], 
    tests_require=['pytest==7.3.1'],
    test_suite='tests',
)
    