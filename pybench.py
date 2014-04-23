from argparse import ArgumentParser
from collections import defaultdict
from contextlib import contextmanager
from cProfile import Profile
from inspect import getfile
from itertools import product
import json
from os import path, makedirs
from pprint import pprint
from subprocess import call
import time

import matplotlib as mpl
mpl.use("Agg")
import pylab


class Benchmark(object):
    """An abstract base class for benchmarks."""
    params = []
    repeats = 3
    warmups = 1
    average = min
    method = 'test'
    timer = time.time
    plotstyle = {}
    profilegraph = {}
    meta = {}
    series = {}

    def __init__(self, **kwargs):
        self.basedir = path.dirname(getfile(self.__class__))
        self.profiledir = path.join(self.basedir, 'profiles')
        self.resultsdir = path.join(self.basedir, 'results')
        self.name = self.__class__.__name__
        if self.series:
            suff = '_'.join('%s%s' % (k, v) for k, v in self.series.items())
            self.name += '_' + suff
        self.description = self.__doc__
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not path.exists(self.profiledir):
            makedirs(self.profiledir)
        if not path.exists(self.resultsdir):
            makedirs(self.resultsdir)
        if isinstance(self.method, str):
            self.method = getattr(self, self.method, self.method)
        self.regions = defaultdict(float)

    @contextmanager
    def timed_region(self, name):
        t_ = self.timer()
        yield
        self.regions[name] += self.timer() - t_

    def _args(self, kwargs):
        name = kwargs.pop('name', self.name)
        params = kwargs.pop('params', self.params)
        method = kwargs.pop('method', self.method)
        if isinstance(method, str):
            method = getattr(self, method)
        return name, params, method

    def parser(self, **kwargs):
        msg = ' (uses the default file name if no file is given)'
        if kwargs:
            epilog = 'The following defaults are used if not overridden:'
            epilog = '\n'.join([epilog, str(kwargs)])
        else:
            epilog = None
        p = ArgumentParser(description=self.description, epilog=epilog)
        p.add_argument('-r', '--run', action='store_true',
                       default=kwargs.get('run', False),
                       help='run the method with default arguments')
        p.add_argument('-b', '--benchmark', action='store_true',
                       default=kwargs.get('benchmark', False),
                       help='run the benchmark')
        p.add_argument('-s', '--save', nargs='?', metavar='file',
                       default=kwargs.get('save', False),
                       help='save results to file' + msg)
        p.add_argument('-l', '--load', nargs='?', metavar='file',
                       default=kwargs.get('load', False),
                       help='load results from file' + msg)
        p.add_argument('-c', '--combine', nargs=1, type=json.loads,
                       metavar='dictionary', default=kwargs.get('combine'),
                       help='combine several results (expects a dictionary of result file name / prefix pairs)')
        p.add_argument('-p', '--plot', type=str, nargs='+', metavar='xaxis',
                       default=kwargs.get('plot'),
                       help='Plot results with given parameter on the x-axis')
        p.add_argument('--profile', action='store_true',
                       default=kwargs.get('profile', False),
                       help='Create a cProfile')
        return p

    def main(self, **kwargs):
        args = self.parser(**kwargs).parse_args()
        if args.run:
            self.method()
        if args.benchmark:
            self.run()
        if args.save or args.save is None:
            self.save(args.save)
        if args.load or args.load is None:
            self.load(args.load)
        if args.combine:
            self.combine(args.combine)
        if args.plot:
            for xaxis in args.plot:
                self.plot(xaxis)
        if args.profile:
            self.profile()

    def profile(self, **kwargs):
        name, params, method = self._args(kwargs)
        profiledir = kwargs.pop('profiledir', self.profiledir)
        profilegraph = kwargs.pop('profilegraph', self.profilegraph)
        out = path.join(profiledir, name)
        pkeys, pvals = zip(*params)
        for pvalues in product(*pvals):
            kargs = dict(zip(pkeys, pvalues))
            suff = '_'.join('%s%s' % (k, v) for k, v in kargs.items())
            pr = Profile()
            pr.runcall(method, **kargs)
            statfile = '%s_%s.pstats' % (out, suff)
            pr.dump_stats(statfile)
            if profilegraph:
                n = profilegraph.get('node_threshold', 1.0)
                e = profilegraph.get('edge_threshold', 0.2)
                for fmt in profilegraph['format'].split(','):
                    graph = '%s_%s.%s' % (out, suff, fmt)
                    cmd = 'gprof2dot -f pstats -n %s -e %s %s | dot -T%s -o %s'
                    call(cmd % (n, e, statfile, fmt, graph), shell=True)

    def run(self, **kwargs):
        name, params, method = self._args(kwargs)
        description = kwargs.pop('description', self.description)
        repeats = kwargs.pop('repeats', self.repeats)
        warmups = kwargs.pop('warmups', self.warmups)
        average = kwargs.pop('average', self.average)

        timings = {}
        pkeys, pvals = zip(*params)
        for pvalues in product(*pvals):
            kargs = dict(zip(pkeys, pvalues))

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
                       'regions': self.regions.keys(),
                       'plotstyle': self.plotstyle,
                       'meta': self.meta,
                       'series': self.series,
                       'timings': timings}
        return self.result

    def save(self, filename=None):
        filename = filename or path.join(self.resultsdir, self.name + '.dat')
        with open(filename, 'w') as f:
            pprint(self.result, f)

    def load(self, filename=None):
        filename = filename or path.join(self.resultsdir, self.name + '.dat')
        with open(filename) as f:
            self.result = eval(f.read())
        return self.result

    def combine(self, files):
        result = {'name': self.name,
                  'description': self.description,
                  'meta': self.meta,
                  'series': self.series,
                  'params': self.params}
        plotstyle = {}
        timings = defaultdict(dict)
        regions = set()
        for name, pref in files.items():
            if path.exists(path.join(self.resultsdir, name + '.dat')):
                filename = path.join(self.resultsdir, name + '.dat')
            else:
                filename = name
            with open(filename) as f:
                res = eval(f.read())
                for k, v in res['plotstyle'].items():
                    plotstyle[pref + ' ' + k] = v
                for k, v in res['timings'].items():
                    # Parametrized benchmark
                    if isinstance(v, dict):
                        for r, t in v.items():
                            timings[k][pref + ' ' + r] = t
                            regions.add(pref + ' ' + r)
                    # Non-parametrized benchmark
                    else:
                        timings[pref + ' ' + k] = v
                        regions.add(pref + ' ' + k)
        result['plotstyle'] = plotstyle
        result['timings'] = timings
        result['regions'] = list(regions)
        self.result = result
        return result

    def combine_series(self, series, filename=None):
        filename = filename or self.name
        self.params = self.params + series
        result = {'name': self.name, 'params': self.params}
        timings = {}
        skeys, svals = zip(*series)
        for svalues in product(*svals):
            suff = '_'.join('%s%s' % (k, v) for k, v in zip(skeys, svalues))
            fname = '%s_%s.dat' % (filename, suff)
            if path.exists(path.join(self.resultsdir, fname)):
                fname = path.join(self.resultsdir, fname)
            with open(fname) as f:
                res = eval(f.read())
                for key in ['description', 'plotstyle', 'meta', 'regions']:
                    result[key] = res[key]
                for k, v in res['timings'].items():
                    timings[k + svalues] = v
        result['timings'] = timings
        self.result = result
        return result

    def plot(self, xaxis, **kwargs):
        timings = kwargs.pop('timings', self.result['timings'])
        figname = kwargs.pop('figname', self.name)
        params = kwargs.pop('params', self.params)
        legend_pos = kwargs.pop('legend_pos', 'best')
        ylabel = kwargs.pop('ylabel', 'time [sec]')
        regions = kwargs.pop('regions', self.result['regions'])
        title = kwargs.pop('title', self.name)
        format = kwargs.pop('format', 'svg')
        plotdir = kwargs.pop('plotdir', self.resultsdir)
        plotstyle = kwargs.pop('plotstyle', self.result['plotstyle'])
        if not path.exists(plotdir):
            makedirs(plotdir)

        pkeys, pvals = zip(*params)
        pnames = [p for p in pkeys if p != xaxis]
        idx = pkeys.index(xaxis)
        xvals = pvals[idx]
        for pv in product(*[p for p in pvals if p != xvals]):
            fsuff = '_'.join('%s%s' % (k, v) for k, v in zip(pnames, pv))
            tsuff = ', '.join('%s=%s' % (k, v) for k, v in zip(pnames, pv))
            fig = pylab.figure(figname + '_' + fsuff, figsize=(8, 6), dpi=300)
            for r in regions:
                yvals = [timings[pv[:idx] + (v,) + pv[idx:]][r] for v in xvals]
                pylab.plot(xvals, yvals, label=r, **plotstyle.get(r, {}))
            pylab.legend(loc=legend_pos)
            pylab.xlabel(xaxis)
            pylab.ylabel(ylabel)
            pylab.title(title + ': ' + tsuff)
            pylab.grid()
            if not format:
                pylab.show()
            else:
                for fmt in format.split(','):
                    pylab.savefig(path.join(plotdir, '%s_%s.%s' % (figname, fsuff, fmt)),
                                  orientation='landscape', format=fmt, transparent=True)
            pylab.close(fig)
