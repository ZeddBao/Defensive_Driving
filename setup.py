from setuptools import setup
from Cython.Build import cythonize
import numpy

# 获取 NumPy 的头文件目录
numpy_include_dir = numpy.get_include()

setup(
    ext_modules=cythonize("carla_utils.pyx", annotate=True),
    include_dirs=[numpy_include_dir]
)