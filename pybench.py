try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
from collections import defaultdict
from contextlib import contextmanager
from itertools import product
import time


class Benchmark(object):
    """An abstract base class for benchmarks."""
    params = {}
    repeats = 3
    warmups = 1
    average = min
    method = 'test'
    timer = time.time

    @contextmanager
    def timed_region(self, name):
        t_ = self.timer()
        yield
        self.regions[name] += self.timer() - t_

    def run(self, **kwargs):
        name = kwargs.pop('name', self.__class__.__name__)
        description = kwargs.pop('description', self.__doc__)
        params = kwargs.pop('params', self.params)
        repeats = kwargs.pop('repeats', self.repeats)
        warmups = kwargs.pop('warmups', self.warmups)
        average = kwargs.pop('average', self.average)
        method = kwargs.pop('method', self.method)
        if isinstance(method, str):
            method = getattr(self, method)

        timings = {}
        self.regions = defaultdict(float)
        for pvalues in product(*params.values()):
            kargs = OrderedDict(zip(params.keys(), pvalues))

            for _ in range(warmups):
                method(**kargs)

            def bench():
                self.regions = defaultdict(float)
                with self.timed_region('total'):
                    method(**kargs)
                return self.regions
            times = [bench() for _ in range(repeats)]
            # Average over all timed regions
            times = dict((k, average(d[k] for d in times))
                         for k in self.regions.keys())
            if pvalues:
                timings[pvalues] = times
            else:
                timings = times
        self.result = {'name': name,
                       'description': description,
                       'params': params,
                       'repeats': repeats,
                       'warmups': warmups,
                       'average': average.__name__,
                       'method': method.__name__,
                       'timings': timings}
        return self.result
