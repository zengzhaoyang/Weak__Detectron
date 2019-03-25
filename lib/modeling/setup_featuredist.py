from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    name='featuredist_cal',
    ext_modules=cythonize([
        Extension('featuredist_cal', ['featuredist_cal.pyx'])   
    ])
)
