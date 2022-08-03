"""A setuptools module for kudos.
See:
https://packaging.python.org/en/latest/distributing.html
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
#with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    #long_description = f.read()

setup(
    name='kudos',
    version='0.0.1',
    description='Tools for optimizing public health case definitions',
    #long_description=long_description,
    #long_description_content_type='text/markdown',
    url='https://github.com/scotthlee/kudos',
    author='Scott Lee',
    author_email='yle4@cdc.gov',
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Epidemiologists',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='combinatorial optimization',
    packages=['kudos'],
    install_requires=[
        'matplotlib', 'numpy', 'ortools',
        'pandas', 'scikit-learn', 'scipy',
        'seaborn', 'tqdm'
    ],
)