from setuptools import setup

setup(name='pybench',
      version='0.0.1',
      description='Benchmarking infrastructure for scientific codes',
      long_description=open('README.rst').read(),
      author='Florian Rathgeber',
      author_email='florian.rathgeber@gmail.com',
      url='https://github.com/firedrakeproject/pybench',
      license='License :: OSI Approved :: BSD License',
      setup_requires=['pytest-runner'],
      tests_require=['pytest', 'pytest-flake8'],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: C',
          'Programming Language :: Cython',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.2',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5'
      ],
      py_modules=['pybench'])
