from distutils.core import setup
from Cython.Build import cythonize
from distutils.core import Extension
import numpy
from teHmm.common import checkRequirements

checkRequirements()

setup(
  name = 'teHmm',
  ext_modules = cythonize([
        Extension("_basehmm", ["_basehmm.pyx"],
                include_dirs=[numpy.get_include()]),
        Extension("_hmm", ["_hmm.pyx"],
                include_dirs=[numpy.get_include()]),
        Extension("_emission", ["_emission.pyx"],
                include_dirs=[numpy.get_include()]),
        Extension("_track", ["_track.pyx"],
                include_dirs=[numpy.get_include()]),                
        Extension("_cfg", ["_cfg.pyx"],
                include_dirs=[numpy.get_include()],
                #extra_compile_args=['-fopenmp'],
                #extra_link_args=['-fopenmp']
                 )
        ])
)
