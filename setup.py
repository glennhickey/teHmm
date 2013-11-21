from distutils.core import setup
from Cython.Build import cythonize
from distutils.core import Extension
import numpy

setup(
  name = 'Hello world app',
  ext_modules = cythonize([Extension("_emission", ["_emission.pyx"],
                                     include_dirs=[numpy.get_include()])])
)

#Cython.Build import cythonize
