from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

setup(
    ext_modules = cythonize("com/uva/sample_latent_vars.pyx")
)
