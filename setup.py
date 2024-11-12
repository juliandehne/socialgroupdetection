# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name="socialgroupdetection",
    version="0.0.1",
    description="a python library to detect social groups mentioned in political texts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/juliandehne/socialgroupdetection",
    author="Julian Dehne",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache 2.0 License",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    include_package_data=True,
    install_requires=[
        "numpy==1.22.3",
        "pandas~=1.5.3",
        "tqdm",
        "jupyter"
    ],
    python_requires='>=3.9, <3.10',  # Allows all Python 3.9.x versions
    setup_requires=[
        "setuptools>=64"
    ],
)
