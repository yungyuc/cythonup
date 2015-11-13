from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules=[
    Extension("cm",
              sources=["cm.pyx"],
              libraries=["m"],
              extra_compile_args=["-w"],
             )
]

setup(
  name = "C_module",
  ext_modules = cythonize(ext_modules)
)

