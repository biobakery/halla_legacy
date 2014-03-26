##### halla setup.py
from setuptools import setup, find_packages
import multiprocessing
setup(
    name = "halla",
    version = "1.0.1",
    license = "MIT", 
    description = "HAllA is a programmatic tool for performing multiple association testing between two or more heterogeneous datasets, each containing a mixture of discrete, binary, or continuous data.",
    author = ["Yo Sup Moon","Curtis Huttenhower"],
    author_email = "moon.yosup@gmail.com",
    url = "http://huttenhower.sph.harvard.edu/halla",
    download_url = "http://huttenhower.sph.harvard.edu/halla",
    keywords = ["association", "testing" ],
    classifiers = [
        "Programming Language :: Python",
        "Development Status :: 1 - Development",
        "Environment :: Other Environment",
        "License :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multiple Association Testing",
        ],
    long_description = open('README.txt').read(),
    install_requires=[
        "Numpy >= 1.6.2",
        "Scipy >= 0.10.1",
        "Matplotlib >= 1.1.1",
        "Scikit-learn  >= 0.14.1"
    ],
    package_data={"testdata": ['halla/testdata',]},
    packages=find_packages(),
    entry_points = {
        'console_scripts': [
            'halla = halla.cli:_main',
        ],
        }
 )
