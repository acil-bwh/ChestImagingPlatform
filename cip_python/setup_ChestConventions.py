import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

cip_dir = '/Users/jross/Downloads/ChestImagingPlatformPrivate/'
cip_build_dir = '/Users/jross/Downloads/ChestImagingPlatformPrivate-superbuild/'

os.environ["CC"] = "/usr/bin/cc"
os.environ["CXX"] = "/usr/bin/c++"

sources = ['wrap_ChestConventions.pyx']
libraries = ["ChestConventions"]
library_dirs = [cip_build_dir + 'lib/']
runtime_library_dirs = library_dirs
include_dirs = [cip_dir + 'Common/']

setup(
    cmdclass = {'build_ext': build_ext},    
    ext_modules = [
        Extension("ChestConventions",
                  sources=sources,
                  include_dirs=include_dirs,
                  language="c++",
                  library_dirs=library_dirs,
                  runtime_library_dirs=runtime_library_dirs,
                  libraries=libraries   
                  )]
)
