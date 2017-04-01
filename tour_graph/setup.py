from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
  name = 'tour_graph',
  ext_modules = cythonize("tour_graph.pyx"),
  include_dirs=[np.get_include()]
)
