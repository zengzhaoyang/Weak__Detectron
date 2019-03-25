from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    name='kmeans',
    ext_modules=cythonize([
        Extension('kmeans', ['kmeans.pyx'])   
    ])
)
