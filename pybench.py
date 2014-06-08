from argparse import ArgumentParser
from collections import defaultdict
from contextlib import contextmanager
from cProfile import Profile
from datetime import datetime
from inspect import getfile
from itertools import product
import json
from os import path, makedirs, remove
from pprint import pprint
import shutil
from subprocess import call
import time

import matplotlib as mpl
mpl.use("Agg")
from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('small')
import numpy as np
import pylab

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
except ImportError:
    # Assume serial
    rank = 0


class Benchmark(object):
    """An abstract base class for benchmarks."""
    params = []
    """The parameters to run the benchmark for, a list of tuples."""
    repeats = 3
    """How often to repeat each benchmark."""
    warmups = 1
    """How man dry runs to perform before timing."""
    average = min
    """The function used to average over multiple benchmark runs."""
    method = 'test'
    """The methods to run the benchmark for."""
    timer = time.time
    """The timer to use."""
    plotstyle = {}
    """The plot style to use for each timed region (a nested dictionary with a
    key per region and a dictionary of plot options as the value)."""
    colormap = 'Set2'
    """The matplotlib colormap to cycle through."""
    profilegraph = {}
    """Options for creating the profile graph with gprof2dot."""
    profileregions = ['total']
    """Regions to create profile graphs for."""
    meta = {}
    """Metadata to include in result output."""
    series = {}
    """Benchmark series created from several invocations of the script e.g.
    for parallel runs on variable number of processors."""
    suffix = '.dat'
    """Suffix for the result file to write."""

    def __init__(self, **kwargs):
        self.basedir = path.dirname(getfile(self.__class__))
        self.plotdir = path.join(self.basedir, 'plots')
        self.profiledir = path.join(self.basedir, 'profiles')
        self.resultsdir = path.join(self.basedir, 'results')
        self.name = getattr(self, 'name', self.__class__.__name__)
        if self.series:
            suff = '_'.join('%s%s' % (k, v) for k, v in self.series.items())
            self.name += '_' + suff
        self.description = self.__doc__
        for k, v in kwargs.items():
            setattr(self, k, v)
        if isinstance(self.method, str):
            self.method = getattr(self, self.method, self.method)
        self.regions = defaultdict(float)
        self.profiles = {}

    @contextmanager
    def timed_region(self, name, normalize=1.0):
        if name in self.profiles:
            self.profiles[name].enable()
        t_ = self.timer()
        yield
        self.regions[name] += (self.timer() - t_) * normalize
        if name in self.profiles:
            self.profiles[name].disable()

    def register_timing(self, name, value):
        self.regions[name] += value

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
        n = profilegraph.get('node_threshold', 1.0)
        e = profilegraph.get('edge_threshold', 0.2)
        formats = profilegraph['format'].split(',') if profilegraph else []
        regions = kwargs.pop('regions', self.profileregions)
        out = path.join(profiledir, name)
        if not path.exists(profiledir):
            makedirs(profiledir)
        pkeys, pvals = zip(*params)
        for pvalues in product(*pvals):
            if rank == 0:
                print 'Profile', name, 'for parameters', ', '.join('%s=%s' % (k, v) for k, v in zip(pkeys, pvalues))
            kargs = dict(zip(pkeys, pvalues))
            suff = '_'.join('%s%s' % (k, v) for k, v in kargs.items())
            for r in regions:
                self.profiles[r] = Profile()
            if 'total' in regions:
                self.profiles['total'].runcall(method, **kargs)
            else:
                method(**kargs)
            if rank == 0:
                for r in regions:
                    statfile = '%s_%s_%s' % (out, suff, r.replace(' ', ''))
                    self.profiles[r].dump_stats(statfile + '.pstats')
                    for fmt in formats:
                        cmd = 'gprof2dot -f pstats -n %s -e %s %s.pstats | dot -T%s -o %s.%s'
                        call(cmd % (n, e, statfile, fmt, statfile, fmt), shell=True)

    def run(self, **kwargs):
        name, params, method = self._args(kwargs)
        description = kwargs.pop('description', self.description)
        repeats = kwargs.pop('repeats', self.repeats)
        warmups = kwargs.pop('warmups', self.warmups)
        average = kwargs.pop('average', self.average)

        timings = {}
        self.result = {'name': name,
                       'description': description,
                       'params': params,
                       'repeats': repeats,
                       'warmups': warmups,
                       'average': average.__name__,
                       'method': method.__name__,
                       'regions': self.regions.keys(),
                       'plotstyle': self.plotstyle,
                       'colormap': self.colormap,
                       'meta': self.meta,
                       'series': self.series,
                       'timings': timings}
        if params:
            pkeys, pvals = zip(*params)
        else:
            pkeys, pvals = (), ()
        for pvalues in product(*pvals):
            if rank == 0:
                print 'Benchmark', name, 'for parameters', ', '.join('%s=%s' % (k, v) for k, v in zip(pkeys, pvalues))
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
            # Auto save
            self.save(suffix='.autosave~')
        return self.result

    def _file(self, filename=None, suffix=None):
        filename = filename or self.name
        suffix = suffix or self.suffix
        if filename.endswith(suffix):
            return filename
        if not path.exists(self.resultsdir):
            makedirs(self.resultsdir)
        return path.join(self.resultsdir, filename + suffix)

    def _read(self, filename=None, suffix=None):
        with open(self._file(filename, suffix)) as f:
            return eval(f.read())

    def load(self, filename=None, suffix=None):
        self.result = self._read(filename)
        return self.result

    def save(self, filename=None, suffix=None):
        if rank > 0:
            return
        if path.exists(self._file(filename, '.autosave~')):
            try:
                remove(self._file(filename, '.autosave~'))
            except OSError:
                pass
        with open(self._file(filename, suffix), 'w') as f:
            pprint(self.result, f)

    def combine(self, files):
        result = {'name': self.name, 'series': self.series}
        plotstyle = {}
        timings = defaultdict(dict)
        regions = set()
        for name, pref in files.items():
            res = self._read(name)
            for key in ['description', 'meta', 'params', 'colormap']:
                result[key] = res[key]
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
            fname = '%s_%s' % (filename, suff)
            res = self._read(fname)
            for key in ['description', 'plotstyle', 'meta', 'regions', 'colormap']:
                result[key] = res[key]
            for k, v in res['timings'].items():
                timings[k + svalues] = v
        result['timings'] = timings
        self.result = result
        return result

    def plot(self, xaxis, **kwargs):
        if rank > 0:
            return
        timings = kwargs.pop('timings', self.result['timings'])
        figname = kwargs.pop('figname', self.result['name'])
        params = kwargs.pop('params', self.result['params'])
        xlabel = kwargs.pop('xlabel', xaxis)
        xvalues = kwargs.pop('xvalues', None)
        ylabel = kwargs.pop('ylabel', 'time [sec]')
        regions = kwargs.pop('regions', self.result['regions'])
        groups = kwargs.get('groups', [])
        title = kwargs.pop('title', self.name)
        legend = kwargs.get('legend', {'loc': 'best'})
        format = kwargs.pop('format', 'svg')
        plotdir = kwargs.pop('plotdir', self.plotdir)
        plotstyle = kwargs.pop('plotstyle', self.result['plotstyle'])
        kinds = kwargs.pop('kinds', 'plot')
        wscale = kwargs.pop('wscale', 0.8)
        bargroups = kwargs.get('bargroups', [''])
        # A tuple of either the same length as groups (speedup relative to a
        # a specimen in the group) or 1 + length of groups (speedup relative to
        # a single data point)
        speedup = kwargs.get('speedup', False)
        transform = kwargs.get('transform')
        # Set the default color cycle according to the given color map
        colormap = kwargs.pop('colormap', self.result.get('colormap', self.colormap))
        cmap = mpl.cm.get_cmap(name=colormap)
        linestyles = ('solid', 'dashed', 'dashdot', 'dotted')
        fillstyles = ('', '/', '\\', '-')
        if not path.exists(plotdir):
            makedirs(plotdir)

        def group(r):
            for i, g in enumerate(bargroups):
                if g in r:
                    return i
            return 0

        pkeys, pvals = zip(*params)
        idx = [pkeys.index(a) for a in [xaxis] + groups]
        pkeys = [p for p in pkeys if p not in [xaxis] + groups]
        xvals = pvals[idx[0]]
        gvals = [pvals[i] for i in idx[1:]]
        pvals = [p for i, p in enumerate(pvals) if i not in idx]
        colors = [cmap(i) for i in np.linspace(0, 0.9, len(regions))]
        mpl.rcParams['axes.color_cycle'] = colors
        speedup_group = speedup and len(speedup) <= len(gvals)
        speedup_single = speedup and len(speedup) == len(gvals) + 1
        if speedup_group:
            for i, s in enumerate(speedup):
                gvals[i] = filter(lambda x: x != s, gvals[i])
        nlines = len(regions) * int(np.prod([len(g) for g in gvals]))

        def lookup(pv, *args):
            for i, a in sorted(zip(idx, args)):
                pv = pv[:i] + (a,) + pv[i:]
            return timings[pv]

        offset = np.arange(len(xvals)) + 0.1
        xticks = np.arange(len(xvals)) + 0.5
        outline = []
        for pv in product(*pvals):
            fsuff = '_'.join('%s%s' % (k, v) for k, v in zip(pkeys, pv))
            if speedup:
                fsuff += '_speedup'
            tsuff = ', '.join('%s=%s' % (k, v) for k, v in zip(pkeys, pv))
            outline += ['<tr>']
            for kind in kinds.split(','):
                fig = pylab.figure(figname + '_' + fsuff, figsize=(9, 6), dpi=300)
                ax = pylab.subplot(111)
                if kind == 'barstacked':
                    ystack = [np.zeros_like(xvals, dtype=np.float) for _ in bargroups]
                plot = {'bar': ax.bar,
                        'barstacked': ax.bar,
                        'barlog': ax.bar,
                        'barstackedlog': ax.bar,
                        'plot': ax.plot,
                        'semilogx': ax.semilogx,
                        'semilogy': ax.semilogy,
                        'loglog': ax.loglog}[kind]
                w = 0.8 / nlines if kind in ['bar', 'barlog'] else len(bargroups)
                i = 0
                for g, gv in enumerate(product(*gvals)):
                    for ir, r in enumerate(regions):
                        try:
                            yvals = np.array([lookup(pv, v, *gv)[r] for v in xvals])
                            label = ', '.join([r] + map(str, gv[len(speedup or []):]))
                            # 1) speedup relative to a specimen in the group
                            if speedup_group:
                                yvals = np.array([lookup(pv, v, *(speedup + gv[len(speedup):]))[r] for v in xvals]) / yvals
                            # 2) speedup relative to a single datapoint
                            elif speedup_single:
                                yvals = lookup(pv, *speedup)[r] / yvals
                            if transform:
                                yvals = transform(xvals, yvals)
                            if kind in ['barstacked', 'barstackedlog']:
                                plot(offset + group(r) * w, yvals, w,
                                     bottom=ystack[group(r)], label=label,
                                     color=colors[ir], hatch=fillstyles[g % 4],
                                     log=kind == 'barstackedlog')
                                pylab.xticks(xticks, xvalues or xvals)
                                ystack[group(r)] += yvals
                            elif kind in ['bar', 'barlog']:
                                plot(offset + i * w, yvals, w, label=label,
                                     color=colors[ir], hatch=fillstyles[g % 4],
                                     log=kind == 'barlog')
                                pylab.xticks(xticks, xvalues or xvals)
                            else:
                                plot(xvalues or xvals, yvals, label=label,
                                     linestyle=linestyles[g % 4], **plotstyle.get(r, {}))
                        except KeyError:
                            raise
                        i += 1
                if legend.get('loc') == 'center left':
                    # Shink current axis by 20%
                    box = ax.get_position()
                    ax.set_position([box.x0, box.y0, box.width * wscale, box.height])
                    # Put a legend to the right of the current axis
                    ax.legend(bbox_to_anchor=(1, 0.5), prop=fontP, **legend)
                else:
                    ax.legend(prop=fontP, **legend)
                pylab.xlabel(xlabel)
                pylab.ylabel(ylabel)
                pylab.title(title + ': ' + tsuff)
                pylab.grid()
                if not format:
                    pylab.show()
                else:
                    for fmt in format.split(','):
                        fname = '%s_%s_%s.%s' % (figname, kind, fsuff, fmt)
                        pylab.savefig(path.join(plotdir, fname),
                                      orientation='landscape', format=fmt,
                                      transparent=True)
                        outline += ['<td><img src="%s"></td>' % fname]
                pylab.close(fig)
            outline += ['</tr>']
            fname = '%s_%s%s.html' % (figname, xaxis, '_speedup' if speedup else '')
            with open(path.join(plotdir, fname), 'w') as f:
                f.write('\n'.join(outline))

    def archive(self, dirname=None):
        timestamp = datetime.now().strftime('%Y-%m-%dT%H%M%S')
        dirname = dirname or path.join(self.basedir, timestamp)
        makedirs(dirname)
        for d in [self.resultsdir, self.profiledir, self.plotdir]:
            shutil.move(d, dirname)
