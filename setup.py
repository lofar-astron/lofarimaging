#!/usr/bin/env python

from setuptools import setup
from lofarimaging import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

if __name__ == "__main__":
    setup(name='lofarimaging',
          version=__version__,
          description='LOFAR imaging utilities',
          long_description=long_description,
          long_description_content_type="text/markdown",
          author='Vanessa Moss, Michiel Brentjens, Tammo Jan Dijkema, Maaijke Mevius (ASTRON)',
          author_email='moss@astron.nl',
          packages=['lofarimaging'],
          url="https://github.com/lofar-astron/lofarimaging",
          requires=['numpy', 'numexpr', 'numba', 'astropy', 'lofargeotiff', 'lofarantpos',
                    'matplotlib', 'folium', 'mercantile', 'owslib', 'packaging', 'Pillow'],
          scripts=[],
          classifiers=[
              "Programming Language :: Python :: 3",
              "License :: OSI Approved :: Apache Software License",
              "Operating System :: OS Independent",
          ]
          )
