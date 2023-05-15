from setuptools import find_packages, setup
setup(
    name='robustify',
    packages=find_packages(include=['robustify', 'tests']),
    version='0.0.3',
    readme = "README.md",
    description = "Robustify:" ,
    long_description="A python library focused on evaluating the effects of adding structurally conserving noise to data. The goal is to provide a comprehensive set of tools for researchers and practitioners interested in exploring the impact of noise on the score and robustness of their machine learning models.",
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
    
