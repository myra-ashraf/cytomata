# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages
import cytomata


def read(*names):
    values = {}
    extensions = ['.txt', '.rst', '.md']
    for name in names:
        value = ''
        for extension in extensions:
            filename = name + extension
            if os.path.isfile(filename):
                value = open(filename).read()
                break
        values[name] = value
    return values


long_description = """
%(README)s
""" % read('README')

setup(
    name=cytomata.__name__,
    version=cytomata.__version__,
    description=cytomata.__description__,
    long_description=long_description,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps",
    ],
    keywords=cytomata.__keywords__,
    author=cytomata.__author__,
    author_email=cytomata.__email__,
    maintainer=cytomata.__author__,
    maintainer_email=cytomata.__email__,
    url=cytomata.__website__,
    license=cytomata.__license__,
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'numpy', 'tensorflow', 'pygame', 'pymunk', 'gym', 'baselines', 'dill'
    ]
)
