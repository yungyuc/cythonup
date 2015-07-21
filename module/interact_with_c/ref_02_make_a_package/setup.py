from __future__ import absolute_import, division, print_function

from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize
import numpy as np


def main():
    extensions = cythonize([
        Extension("lax.march_c",
                  ["lax/march_c.pyx", "lax/_march_c.c"],
                  include_dirs=[np.get_include()]),
    ])
    setup(
        packages=["lax"],
        ext_modules=extensions,
    )


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 nobomb ai et sw=4 ts=4 tw=79:
