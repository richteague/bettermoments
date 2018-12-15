#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from distutils.core import setup


if sys.version < '3.3':
    raise RuntimeError("bettermoments needs Python 3.3 or newer.")

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md")) as f:
    long_description = f.read()

setup(
    name="bettermoments",
    version="1.1.0",
    author="Richard Teague & Daniel Foreman-Mackey",
    author_email='rteague@umich.edu',
    packages=['bettermoments', 'bettermoments.tests'],
    url="https://github.com/richteague/bettermoments",
    license="LICENSE.md",
    description=("Robust moment map making."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["numpy", "astropy", "argparse"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'bettermoments = bettermoments.collapse_cube:main',
        ],
    }
)
