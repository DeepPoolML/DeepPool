from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='deeppool_bench',
      ext_modules=[cpp_extension.CppExtension(
          name='deeppool_bench', sources=['bench.cpp'], extra_compile_args=['-g']
      )],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
