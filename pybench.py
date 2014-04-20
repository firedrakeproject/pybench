try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
from collections import defaultdict
from contextlib import contextmanager
from itertools import product
import time


class Benchmark(object):
    params = {}
    repeats = 3
    warmups = 1
    average = min
    method = 'test'
    name = 'Benchmark'
    timer = time.time

    @contextmanager
    def timed_region(self, name):
        t_ = self.timer()
        yield
        self.regions[name] += self.timer() - t_

    def run(self, **kwargs):
        params = kwargs.pop('params', self.params)
        repeats = kwargs.pop('repeats', self.repeats)
        warmups = kwargs.pop('warmups', self.warmups)
        average = kwargs.pop('average', self.average)
        method = kwargs.pop('method', self.method)
        name = kwargs.pop('name', self.name)
        if isinstance(method, str):
            method = getattr(self, method)

        timings = {}
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
            if times:
                # Average over all timed regions
                times = dict((k, average(d[k] for d in times))
                             for k in self.regions.keys())
                if pvalues:
                    timings[pvalues] = times
                else:
                    timings = times
        return {'name': name, 'timings': timings}
