from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    name='iou_cal',
    ext_modules=cythonize([
        Extension('iou_cal', ['iou_cal.pyx'])   
    ])
)
