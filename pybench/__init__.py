from os import path

from benchmark import Benchmark, parser, timed  # NOQA: export

# Use README as module documentation
readme = path.join(path.dirname(__file__), '..', 'README.rst')
if path.exists(readme):
    with open(readme) as f:
        __doc__ = f.read()
del readme
del f
