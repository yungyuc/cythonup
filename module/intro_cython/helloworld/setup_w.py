from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

extensions = [Extension('helloworld4', ['helloworld.pyx', ],
                        extra_compile_args=["-w"],
                        extra_link_args=[])]

setup(
    ext_modules = cythonize(extensions)
)
