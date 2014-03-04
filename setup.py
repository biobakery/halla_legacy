#  halla setup.py
from distutils.core import setup
setup(
    name = "halla",
    py_modules = ["halla"],
    packages = ["halla", 
    "halla",
    "halla/stats",
    "halla/preset",
    "halla/logger",
    "halla/distance",
    "halla/plot",
    "halla/parser",
    "halla/test",
    "halla/hierarchy"
    ],
    version = "1.0.1",
    description = "HAllA is a programmatic tool for performing multiple association testing between two or more heterogeneous datasets, each containing a mixture of discrete, binary, or continuous data.",
    author = "Yo Sup Moon",
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
    long_description = open('README.txt').read()
)
