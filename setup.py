
try:
    from setuptools import setup, find_packages
except ImportError:
    sys.exit("Please install setuptools.")

import os
import urllib

VERSION = "0.6.15"
AUTHOR = "Gholamali Rahnavard, Curtis Huttenhower, Huttenhower Lab"
AUTHOR_EMAIL = "halla-users@googlegroups.com"
MAINTAINER = "Gholamali Rahnavard"
MAINTAINER_EMAIL = "gholamali.rahnavard@gmail.com"

# try to download the bitbucket counter file to count downloads
# this has been added since PyPI has turned off the download stats
# this will be removed when PyPI Warehouse is production as it
# will have download stats
COUNTER_URL="http://bitbucket.org/biobakery/halla/downloads/counter.txt"
counter_file="counter.txt"
if not os.path.isfile(counter_file):
    print("Downloading counter file to track halla downloads"+
        " since the global PyPI download stats are currently turned off.")
    try:
        file, headers = urllib.urlretrieve(COUNTER_URL,counter_file)
    except EnvironmentError:
        print("Unable to download counter")

setup(
    name="halla",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    version=VERSION,
    license="MIT",
    description="HAllA: Hierarchically All-against-All Association Testing",
    long_description="HAllA is a programmatic tool for performing multiple association testing " + \
        "between two or more heterogeneous datasets, each containing a mixture of discrete, binary, or continuous data." ,
    url="http://huttenhower.sph.harvard.edu/halla",
    keywords=['association','discovery','test','pattern','hierarchically'],
    platforms=['Linux','MacOS'],
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
        ],
    #long_description=open('readme.md').read(),
    install_requires=[  
        "Numpy >= 1.9.2",
        "Scipy >= 0.15.1",
        "Matplotlib >= 1.5.1",
        "Scikit-learn >= 0.14.1",
        #"minepy >= 1.0.0", #for MIC in evaluation package 
        "pandas >= 0.18.1",
        #"R >= 3.1.0",
        #"rpy2 >= 0.0.0"
    ],
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'halla = halla.halla:main',
            'hallagram = halla.hallagram:main',
            'hallaclust = halla.hallaclust:main'
        ]},
    test_suite= 'halla.tests.halla_test.main',
    zip_safe = False
 )
