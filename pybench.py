try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
from collections import defaultdict
from contextlib import contextmanager
from inspect import getfile
from itertools import product
from os import path, makedirs
from pprint import pprint
import time

import matplotlib as mpl
mpl.use("Agg")
import pylab


class Benchmark(object):
    """An abstract base class for benchmarks."""
    params = {}
    repeats = 3
    warmups = 1
    average = min
    method = 'test'
    timer = time.time
    save = True

    def __init__(self, **kwargs):
        self.resultsdir = path.join(path.dirname(getfile(self.__class__)), 'results')
        self.name = self.__class__.__name__
        self.description = self.__doc__
        for k, v in kwargs:
            setattr(self, k, v)
        if not path.exists(self.resultsdir):
            makedirs(self.resultsdir)

    @contextmanager
    def timed_region(self, name):
        t_ = self.timer()
        yield
        self.regions[name] += self.timer() - t_

    def run(self, **kwargs):
        name = kwargs.pop('name', self.name)
        description = kwargs.pop('description', self.description)
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
        if self.save:
            with open(path.join(self.resultsdir, name + '.dat'), 'w') as f:
                pprint(self.result, f)
        return self.result

    def load(self, filename=None):
        filename = filename or path.join(self.resultsdir, self.name + '.dat')
        with open(filename) as f:
            self.result = eval(f.read())

    def plot(self, xaxis, **kwargs):
        timings = kwargs.pop('timings', self.result['timings'])
        figname = kwargs.pop('figname', self.__class__.__name__)
        params = kwargs.pop('params', self.params)
        legend_pos = kwargs.pop('legend_pos', 'best')
        ylabel = kwargs.pop('ylabel', 'time [sec]')
        regions = kwargs.pop('regions', self.regions.keys())
        title = kwargs.pop('title', self.__class__.__name__)
        format = kwargs.pop('format', 'svg')
        plotdir = kwargs.pop('plotdir', self.resultsdir)
        if not path.exists(plotdir):
            makedirs(plotdir)

        pnames = [p for p in params.keys() if p != xaxis]
        idx = params.keys().index(xaxis)
        xvals = params[xaxis]
        for pv in product(*[params[p] for p in pnames]):
            figname += '_' + '_'.join('%s%s' % (k, v) for k, v in zip(pnames, pv))
            title += ', ' + ', '.join('%s=%s' % (k, v) for k, v in zip(pnames, pv))
            fig = pylab.figure(figname, figsize=(8, 6), dpi=300)
            for r in regions:
                yvals = [timings[pv[:idx] + (v,) + pv[idx:]][r] for v in xvals]
                pylab.plot(xvals, yvals, label=r)
            pylab.legend(loc=legend_pos)
            pylab.xlabel(xaxis)
            pylab.ylabel(ylabel)
            pylab.title(title)
            pylab.grid()
            if not format:
                pylab.show()
            else:
                for fmt in format.split(','):
                    pylab.savefig(path.join(plotdir, '%s.%s' % (figname, fmt)),
                                  orientation='landscape', format=fmt, transparent=True)
            pylab.close(fig)
