try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='pybench',
      version='0.0.1',
      description='Benchmarking infrastructure for scientific codes',
      long_description=open('README.rst').read(),
      author='Florian Rathgeber',
      author_email='florian.rathgeber@gmail.com',
      url='https://github.com/firedrakeproject/pybench',
      license='License :: OSI Approved :: BSD License',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: C',
          'Programming Language :: Cython',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
      ],
      py_modules=['pybench'])
