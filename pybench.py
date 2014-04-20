try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
from contextlib import contextmanager
from itertools import product
import time


class Benchmark(object):
    params = {}
    repeats = 3
    warmups = 1
    average = min
    method = None
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
        timer = kwargs.pop('timer', self.timer)

        timings = {}
        for pvalues in product(*params.values()):
            kargs = OrderedDict(zip(params.keys(), pvalues))

            def bench():
                t_ = timer()
                method(**kargs)
                return timer() - t_
            for _ in range(warmups):
                method(**kargs)
            timings[pvalues] = average(bench() for _ in range(repeats))
        return {'name': name, 'timings': timings}
