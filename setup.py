
try:
    from setuptools import setup, find_packages
except ImportError:
    sys.exit("Please install setuptools.")


setup(
    name="halla",
    version="0.6.0",
    license="MIT",
    description="HAllA is a programmatic tool for performing multiple association testing between two or more heterogeneous datasets, each containing a mixture of discrete, binary, or continuous data.",
    author="Gholamali Rahnavard, Curtis Huttenhower",
    author_email="rahnavar@hsph.harvard.edu",
    url="http://huttenhower.sph.harvard.edu/halla",
    keywords=["association", "testing" ],
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 1 - Development",
        "Environment :: Console",
        "License :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
        ],
    long_description=open('readme.md').read(),
    install_requires=[  #### version numbers based on what comes with Anaconda Python, March 26, 2014 
        #"Numpy >= 1.9.2",
        #"Scipy >= 0.13.0",
        "Matplotlib >= 1.1.1",
        "Scikit-learn  >= 0.14.1",
        "minepy >= 1.0.0", #for MIC in evaluation package 
        "pandas >=0.15.2",
        #"R >= 3.1.0",
        "rpy2 >= 0.0.0"
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
