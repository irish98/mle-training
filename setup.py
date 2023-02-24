import os.path as op
from distutils.core import setup

from setuptools import PEP420PackageFinder

ROOT = op.dirname(op.abspath(__file__))
SRC = op.join(ROOT, "src")


setup(
    name="package_housingprice",
    version="1.0.0",
    package_dir={"": "src"},
    description="Housing price prediction",
    author="Rishabh Saxena",
    packages=PEP420PackageFinder.find(where=str(SRC)),
)
