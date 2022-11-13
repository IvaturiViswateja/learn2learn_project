from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

package = Extension('task_dataset', ['task_dataset.pyx'], include_dirs=[numpy.get_include()])
setup(ext_modules=cythonize([package]))