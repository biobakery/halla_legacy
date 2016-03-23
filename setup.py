
try:
    from setuptools import setup, find_packages
except ImportError:
    sys.exit("Please install setuptools.")
VERSION = "0.6.0"
AUTHOR = "Gholamali Rahnavard, Curtis Huttenhower, Huttenhower Lab"
AUTHOR_EMAIL = "halla-users@googlegroups.com"
MAINTAINER = "Gholamali Rahnavard"
MAINTAINER_EMAIL = "gholamali.rahnavard@gmail.com"
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
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
        ],
    #long_description=open('readme.md').read(),
    install_requires=[  
        "Numpy >= 1.9.2",
        "Scipy >= 0.13.0",
        "Matplotlib >= 1.5.1",
        "Scikit-learn >= 0.14.1",
        #"minepy >= 1.0.0", #for MIC in evaluation package 
        "pandas >= 0.15.2",
        #"R >= 3.1.0",
        #"rpy2 >= 0.0.0"
    ],
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'halla = halla.halla:_main',
            'hallagram = halla.hallagram:main'
        ]},
    test_suite= 'halla.tests.halla_test.main',
    zip_safe = False
 )
