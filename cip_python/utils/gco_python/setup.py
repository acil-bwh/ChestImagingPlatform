import os
import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

cip_source_dir = 'C:/Projects/CIP-Source'
cip_library_dir = 'C:/Projects/CIP-Build/lib/Release'
cip_python_dir = 'C:/Projects/CIP-Source/cip_python'

os.environ["CC"] = "/usr/bin/cc"
os.environ["CXX"] = "/usr/bin/c++"
os.environ["MACOSX_DEPLOYMENT_TARGET"] = ""

sources = cip_python_dir + '/utils/gco_python/gco_python.pyx'
libraries = ["GraphCutsOptimization"]
library_dirs = cip_source_dir + '/Utilities/GraphCutsOptimization'
runtime_library_dirs = [] # setting it may cause exception on Windows build
include_dirs = cip_source_dir + '/Utilities/GraphCutsOptimization'

files = ['GCoptimization.cpp', 'graph.cpp', 'LinkedBlockList.cpp',
         'maxflow.cpp']
files = [os.path.join(library_dirs, f) for f in files]
files.insert(0, sources)

setup(
        cmdclass = {'build_ext': build_ext},
        ext_modules = [
            Extension("pygco",
                      files,
                      include_dirs=[include_dirs, numpy.get_include()],
                      language="c++",
                      library_dirs=[library_dirs],
    )]
)
