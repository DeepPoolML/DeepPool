from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='be_training',
      ext_modules=[cpp_extension.CppExtension(
      	'be_training', ['torch_be_training.cpp', 'model.cpp', 'best_effort_training.cpp'],
      	include_dirs=['/usr/local/cuda/targets/x86_64-linux/include/', '/usr/local/cuda-11.0/targets/x86_64-linux/include/'],
      	libraries=['c10_cuda']
      )],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
