from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension("lattice_ray_big11",
              sources=["lattice_ray_big11.pyx"],
              include_dirs=[np.get_include()])  # 添加 NumPy 的头文件路径
]

setup(
    ext_modules=cythonize(ext_modules),
)
