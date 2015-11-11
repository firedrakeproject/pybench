from argparse import ArgumentParser
from collections import defaultdict
from contextlib import contextmanager
from copy import copy
from cProfile import Profile
from datetime import datetime
from functools import wraps
from inspect import getfile
from itertools import product
import json
from os import getenv, path, makedirs
from pprint import pprint
import shutil
from subprocess import call, check_output, CalledProcessError
import time
from warnings import warn

import pandas as pd
import xray

from utils import value_combinations

# Imports for plot, warn if those fail but do not die
try:
    import matplotlib as mpl
    mpl.use("Agg")
    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()
    fontP.set_size('x-small')
    import matplotlib.pyplot as plt
    import numpy as np
except (ImportError, AttributeError):
    warn("Importing matplotlib failed, plot functionality not available.")

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
except ImportError:
    # Assume serial
    rank = 0

html_table = """
<html>
<head>
  <link rel="stylesheet" href="https://github.hubspot.com/sortable/css/sortable-theme-bootstrap.css" />
  <style>
    table[data-sortable] td {
      text-align: right;
    }
  </style>
</head>
<body>
  <table class="sortable-theme-bootstrap" data-sortable>
  %s
  <script src="https://github.hubspot.com/sortable/js/sortable.min.js"></script>
</body>
</html>
"""


def get_git_revision(cwd=None):
    try:
        return check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).strip()
    except CalledProcessError:
        return 'unknown'


def timed(f):
    """A decorator timing a function and returning the execution time alongside
    the result."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        t_ = time.time()
        ret = f(*args, **kwargs)
        return time.time() - t_, ret
    return wrapper


def parser(**kwargs):
    """Build an argument parser for plot scripts."""
    p = ArgumentParser(**kwargs)
    p.add_argument('-i', '--resultsdir',
                   help='directory containing results')
    p.add_argument('-o', '--plotdir',
                   help='directory where to create the plots')
    p.add_argument('-s', '--sequential', action='store_true',
                   help='create plots for sequential runs')
    p.add_argument('-p', '--parallel', type=int, nargs='+', metavar='NP',
                   help='create plots for strong scaling parallel runs')
    p.add_argument('-w', '--weak', type=int, nargs='+', metavar='NP',
                   help='create plots for weak scaling parallel runs')
    return p


class Benchmark(object):
    """An abstract base class for benchmarks."""
    params = {}
    """The parameters to run the benchmark for: a list of pairs, each of which
    is the parameter name and a list of benchmark values."""
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
    """The plot style to use for each timed region: a nested dictionary with a
    key per region and a dictionary of plot options as the value. These are
    passed straight to the matplotlib plot function."""
    colormap = 'Set2'
    """The matplotlib colormap to cycle through."""
    colors = []
    """The colors to cycle through (overrides colormap)."""
    profilegraph = {}
    """Options for creating the profile graph with gprof2dot:
        * node_threshold: eliminate nodes below this threshold
        * edge_threshold: eliminate edges below this threshold
        * format: comma-separated list of output formats (supported by dot)"""
    regions = ['total']
    """Regions to time."""
    profileregions = ['total']
    """Regions to create profile graphs for."""
    meta = {}
    """Metadata to include in result output."""
    result = {}
    """Results data, including timings."""
    series = {}
    """Benchmark series created from several invocations of the script e.g.
    for parallel runs on variable number of processors."""
    suffix = '.nc'
    """Suffix for the result file to write."""

    def __init__(self, **kwargs):
        """:param kwargs: set as attributes on the instance"""
        self.basedir = path.dirname(getfile(self.__class__))
        self.plotdir = path.join(self.basedir, 'plots')
        self.profiledir = path.join(self.basedir, 'profiles')
        self.resultsdir = path.join(self.basedir, 'results')
        self.tabledir = path.join(self.basedir, 'tables')
        self.benchmark = getattr(self, 'benchmark', self.__class__.__name__)
        self.description = self.__doc__
        for k, v in kwargs.items():
            if v is not None:
                setattr(self, k, v)
        if isinstance(self.method, str):
            self.method = getattr(self, self.method, self.method)
        self.profiles = {}
        self.meta['benchmark_version'] = get_git_revision()
        self.meta['pybench_version'] = get_git_revision(path.dirname(__file__))
        if getenv('HOSTNAME'):
            self.meta['hostname'] = getenv('HOSTNAME')
        if getenv('PBS_JOBID'):
            self.meta['jobid'] = getenv('PBS_JOBID')
        if getenv('PBS_JOBNAME'):
            self.meta['jobname'] = getenv('PBS_JOBNAME')
        self.timings = defaultdict(float)
        self.data = self._init_data()

    def _init_data(self, params=None):
        params = dict(params or self.params)
        # Add the regions as another dimension to the results data
        params['region'] = self.regions
        shape = tuple(len(p) for p in params.values())
        array = np.zeros(shape=shape)
        array.fill(np.nan)
        return xray.DataArray(array, coords=params, name=self.benchmark)

    @property
    def name(self):
        """Benchmark name: produced by concatenating the benchmark attribute
        with all keys and values in the series."""
        if self.series:
            suff = '_'.join('%s%s' % (k, v) for k, v in sorted(self.series.items()))
            return self.benchmark + '_' + suff
        return self.benchmark

    @contextmanager
    def timed_region(self, name, normalize=1.0):
        """A context manger for timing a region of code identified by name."""
        if name not in self.regions:
            raise ValueError("%s is not a valid region: %s" % (name, self.regions))
        if name in self.profiles:
            self.profiles[name].enable()
        t_ = self.timer()
        yield
        self.timings[name] += (self.timer() - t_) * normalize
        if name in self.profiles:
            self.profiles[name].disable()

    def register_timing(self, name, value):
        """Register the timing `value` for the region identified by `name`."""
        if name not in self.regions:
            raise ValueError("%s is not a valid region: %s" % (name, self.regions))
        self.timings[name] += value

    def _args(self, kwargs):
        """Parse name, params and method from the kwargs dictionary."""
        name = kwargs.pop('name', self.name)
        params = kwargs.pop('params', self.params)
        method = kwargs.pop('method', self.method)
        if isinstance(method, str):
            method = getattr(self, method)
        return name, params, method

    def parser(self, **kwargs):
        """Return an argument parser, where default values can we overridden
        through `kwargs`:
        * run: defaults to False
        * benchmark: defaults to False
        * save: defaults to False
        * combine: defaults to None
        * plot: defaults to None
        * profile: defaults to False
        """
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
        """Main driver method: parses command line arguments and calls methods
        accordingly.

        :param kwargs: any keyword arguments are forwarded to the parser"""
        args, extra = self.parser(**kwargs).parse_known_args()
        extra = filter(lambda s: '=' in s, extra)
        # Any extra arguments are passed on to the method, but need converting
        # to the right type first
        f = self.method.im_func
        defaults = dict(zip(f.func_code.co_varnames[1:f.func_code.co_argcount], f.func_defaults))
        # Caveat: bool("any_string") is always True, but bool("") is always False
        convert = lambda k, v: (k, v in ['True', 'true'] if isinstance(defaults[k], bool)
                                else type(defaults[k])(v))
        fargs = dict(convert(*a.split('=')) for a in extra)
        if args.load or args.load is None:
            self.load(args.load)
        if args.run:
            self.method(**fargs)
        if args.benchmark:
            self.run(**fargs)
        if args.save or args.save is None:
            self.save(args.save)
        if args.combine:
            self.combine(args.combine)
        if args.plot:
            for xaxis in args.plot:
                self.plot(xaxis)
        if args.profile:
            self.profile(**fargs)

    def profile(self, **kwargs):
        """Create a profile for the given benchmark.

        :param kwargs: keyword arguments override global attributes:
            * name: benchmark name
            * params: benchmark parameters
            * method: method to benchmark
            * profiledir: directory to write profiles to
            * profilegraph: options passed to gprof2dot
            * regions: regions to profile
        """
        name, params, method = self._args(kwargs)
        profiledir = kwargs.pop('profiledir', self.profiledir)
        profilegraph = kwargs.pop('profilegraph', self.profilegraph)
        n = profilegraph.get('node_threshold', 1.0)
        e = profilegraph.get('edge_threshold', 0.2)
        formats = profilegraph['format'].split(',') if profilegraph else []
        regions = kwargs.pop('regions', self.profileregions)
        out = path.join(profiledir, name)
        if rank == 0 and not path.exists(profiledir):
            makedirs(profiledir)
        for param in value_combinations(params):
            if rank == 0:
                print 'Profile', name, 'for parameters', ', '.join('%s=%s' % (k, v) for k, v in sorted(param.items()))
            kwargs.update(param)
            # Dry run
            method(**kwargs)
            suff = '_'.join('%s%s' % (k, v) for k, v in sorted(param.items()))
            for r in regions:
                self.profiles[r] = Profile()
            if 'total' in regions:
                self.profiles['total'].runcall(method, **kwargs)
            else:
                method(**kwargs)
            if rank == 0:
                for r in regions:
                    statfile = '%s_%s_%s' % (out, suff, r.replace(' ', ''))
                    self.profiles[r].dump_stats(statfile + '.pstats')
                    for fmt in formats:
                        cmd = 'gprof2dot -f pstats -n %s -e %s %s.pstats | dot -T%s -o %s.%s'
                        call(cmd % (n, e, statfile, fmt, statfile, fmt), shell=True)

    def run(self, **kwargs):
        """Run the benchmark.

        :param kwargs: keyword arguments override global attributes:
            * name: benchmark name
            * params: benchmark parameters
            * method: method to benchmark
            * description: benchmark description
            * repeats: how often to repeat the benchmark
            * warmups: how many dry run to perform
            * average: method used to average the repeated runs
            * all remaining keyword arguments are passed to the method
        """
        name, params, method = self._args(kwargs)
        description = kwargs.pop('description', self.description)
        repeats = kwargs.pop('repeats', self.repeats)
        warmups = kwargs.pop('warmups', self.warmups)
        average = kwargs.pop('average', self.average)

        # Check that all params are defined at the benchmark level
        for param, vals in params.items():
            if param not in self.params:
                raise ValueError('Unknown parameter ' + param)
            for v in vals:
                if v not in self.params[param]:
                    raise ValueError('Invalid value %s for parameter %s (valid: %s)' %
                                     (v, param, self.params[param]))

        self.meta['start_time'] = str(datetime.now())

        for param in value_combinations(params):
            if rank == 0:
                pstr = ', '.join('%s=%s' % (k, v) for k, v in param.items())
                sstr = ', '.join('%s=%s' % (k, v) for k, v in self.series.items())
                print 'Benchmark', name, 'for parameters', pstr, 'series', sstr
            kwargs.update(param)

            if rank == 0:
                print '  Running', warmups, 'warmup runs'
            for _ in range(warmups):
                method(**kwargs)

            def bench():
                self.timings = defaultdict(float)
                with self.timed_region('total'):
                    method(**kwargs)
                return self.timings
            if rank == 0:
                print '  Running', repeats, 'benchmark runs'
            times = [bench() for _ in range(repeats)]
            # Average over all timings
            if times:
                self.data.loc[param] = [average(d[k] for d in times) for k in self.timings.keys()]
        self.meta['end_time'] = str(datetime.now())
        return self

    def __call__(self, **kwargs):
        return self.data.loc[kwargs]

    def _file(self, filename=None, suffix=None):
        """Return a filepath specified by given `filename` and `suffix`, which
        default to the global name and suffix attributes if not given."""
        filename = filename or self.name
        suffix = suffix or self.suffix
        if filename.endswith(suffix):
            return filename
        if rank == 0 and not path.exists(self.resultsdir):
            makedirs(self.resultsdir)
        return path.join(self.resultsdir, filename + suffix)

    def _read(self, filename=None, suffix=None):
        """Read a netCDF file specified by given `filename` and `suffix`, which
        default to the global name and suffix attributes if not given."""
        return xray.open_dataset(self._file(filename, suffix))

    def load(self, filename=None, suffix=None):
        """Load results from a file specified by given `filename` and `suffix`,
        which default to the global name and suffix attributes if not given."""
        try:
            self.data = self._read(filename)
        except IOError:
            self.data = self._init_data()
        return self

    def save(self, filename=None, suffix=None):
        """Save results to a file specified by given `filename` and `suffix`,
        which default to the global name and suffix attributes if not given."""
        if rank > 0:
            return
        self.data.to_dataset(name=self.name).to_netcdf(self._file(filename, suffix))
        return self

    def combine(self, name, labels, files):
        """Combine results read from the file names in `files` along a new
        dimension `name` with the labels `labels`."""
        arrays = [self._read(f) if isinstance(f, str) else f for f in files]
        self.data = xray.concat(arrays, pd.Index(labels, name=name))
        return self

    def combine_series(self, series, filename=None, coords=None):
        """Combine the results of one or more series of benchmarks.

        :param series: a dictionary with the series names as keys and the list
            of values defing the series as value.
        :param filename: the basename for the files to combine (defaults to the
            global name property if not given)
        :param coords: dictionary of coordinate axes to relabel, where the key
            is the coordinate and the value is the list of new labels
        """
        # Add the series as additional dimensions to the parameter space
        self.params.update(series)
        data = self._init_data()

        filename = filename or self.benchmark

        # Read file for each combination of the series
        for s in value_combinations(series):
            suff = '_'.join('%s%s' % (k, v) for k, v in sorted(s.items()))
            fname = '%s_%s' % (filename, suff)
            data.loc[s] = self._read(fname)[fname]

        # Re-label coordinates if requested
        for k, v in (coords or {}).items():
            data.coords[k] = v

        self.data = data
        return self

    def table(self, **kwargs):
        """Export results as html or latex table (requires pandas).

        :param kwargs: keyword arguments override values given in the results
            * filename: base name of output file
            * dataframe: pandas DataFrame to export (when given, the keyword
              arguments params, regions, skip and timings are ignored)
            * tabledir: output directory
            * regions: regions to output
            * format: comma-separated list of output formats (html, latex, both)
            * precision: precision of floating point values
        """
        if rank > 0:
            return
        filename = kwargs.pop('filename', self.name)
        df = kwargs.get('dataframe') or self.data.to_dataframe()
        if kwargs.get('regions'):
            df = df[kwargs.get('regions')]
        tabledir = kwargs.pop('tabledir', self.tabledir)
        formats = kwargs.pop('format', 'html').split(',')
        if not path.exists(tabledir):
            makedirs(tabledir)
        pd.set_option('display.precision', kwargs.pop('precision', 4))

        # Reset the index only if it is a MultiIndex
        if hasattr(df.index, 'levels'):
            df.reset_index(inplace=True)
        # Pandas is somewhat dumb when it comes to formatting tables, so strip
        # the first line declaring the table and use our own in html_table
        render = {'html': lambda df: html_table % '\n'.join(df.to_html().split('\n')[1:]),
                  'tex': lambda df: df.to_latex(index=False)}
        for fmt in formats:
            with open(path.join(tabledir, "%s.%s" % (filename, fmt)), 'w') as f:
                f.write(render[fmt](df))
        return self

    def lookup(self, region, params, keyset=()):
        """Retrieve a specific timing from benchmark results

        :param region: timed region for which to extract the timing
        :param params: parameter dict or tuple-list used as the key
        :param keyset: list of name-value tuples to specify additional
            lookup parameters
        """
        if isinstance(params, list):
            params = dict(params)
        params.update(keyset)
        pvals = zip(*sorted(params.items()))[1]
        timings = self.result['timings'].get(pvals)
        return timings[region] if timings is not None else np.nan

    def subplot(self, ax, xaxis, kind='plot', **kwargs):
        """Plot a graph into the given axes

        :param ax: the axes to plot into
        :param kind: type of plot
            * bar: bar plot
            * barstacked: stacked bar plot
            * barlog: bar plot with logarithmic y-axis
            * barstackedlog: stacked bar plot with logarithmic y-axis
            * plot: regular plot
            * semilogx: plot with logarithmic x-axis
            * semilogy: plot with logarithmic y-axis
            * loglog: log-log plot
        :param kwargs: keyword arguments override values given in the results
            * axis: if set to "tight", use tight axis (default for subplot)
            * bargroups: for a stacked bar plot, group these parameters next
                to each other instead of stacking them
            * baseline: Add a baseline to the plot: tuple of parameter value
                (needs to be part of groups) and a value along the x-axis, to
                be plotted along the entire length of the axis
            * colormap: color map to cycle through
            * colors: colors to cycle through (overrides colormap)
            * grid: enables grid lines
            * hidexticks: list of indices of xtick labels to hide
            * hideyticks: list of indices of ytick labels to hide
            * hscale: scale factor for height of the plot
            * labels: either a dictionary of one label per group or "compact",
                to generate short labels with only the parameter values, or
                "long", which includes parameter names and values
            * legend: dictionary of legend options, passed as keyword argument
                to the matplotlib legend function
            * lines: additional lines to plot (list of pairs of y-values, where
                scalars are usef for all x-values, and a dict with line styles)
            * linewidth: line width of plots (defaults to 2)
            * plotstyle: plot style to use for each timed region (nested
                dictionary with a key per region and a dictionary of plot
                options as the value, passed straight to the matplotlib plot)
            * regions: regions to plot
            * speedup: tuple of either the same length as groups (speedup
                relative to a specimen in the group) or 1 + length of groups
                (speedup relative to a single data point)
            * ticksize: custom tick label size
            * timings: benchmark timings
            * title: plot title (defaults to the name property)
            * transform: function to transform the y-values (receives x-values
                and y-values as parameters)
            * trendline: Add a trendline for perfect speedup with given label
            * wscale: scale factor for width of the plot
            * xlabel: x-axis label
            * xmax: set maximum of x-axis
            * xmin: set minimum of x-axis
            * xtickbins: number of bins to show along the x axis
            * xticklabels: custom xtick labels (uses xvalues as the x ticks)
            * xvals: values to use for x axis (overrides parameters)
            * xvalues: values to use for x tick labels (defaults to the
                parameter values selected through `xaxis`)
            * ylabel: y-axis label (defaults to "time [sec]")
            * ymax: set maximum of y-axis
            * ymin: set minimum of y-axis
        """
        axis = kwargs.get('axis')
        bargroups = kwargs.get('bargroups', [''])
        baseline = kwargs.get('baseline')
        colormap = kwargs.pop('colormap', self.colormap)
        colors = kwargs.pop('colors', self.colors)
        grid = kwargs.pop('grid', False)
        hidexticks = kwargs.pop('hidexticks', None)
        hideyticks = kwargs.pop('hideyticks', None)
        hscale = kwargs.get('hscale')
        labels = kwargs.get('labels', 'compact')
        legend = kwargs.get('legend', {'loc': 'best'})
        lines = kwargs.get('lines', [])
        linewidth = kwargs.pop('linewidth', 2)
        regions = kwargs.pop('regions', self.result['regions'])
        ticksize = kwargs.get('ticksize')
        timings = kwargs.pop('timings', self.result['timings'])
        title = kwargs.pop('title', self.name)
        transform = kwargs.get('transform')
        xlabel = kwargs.pop('xlabel', None)
        xmax = kwargs.get('xmax')
        xmin = kwargs.get('xmin')
        xtickbins = kwargs.get('xtickbins')
        xticklabels = kwargs.pop('xticklabels', None)
        xvals = kwargs.pop('xvals')
        xvalues = kwargs.pop('xvalues', xvals)
        plotstyle = kwargs.pop('plotstyle', self.plotstyle)
        speedup = kwargs.get('speedup', False)
        trendline = kwargs.get('trendline')
        wscale = kwargs.get('wscale')
        xticks = np.arange(len(xvals)) + 0.5
        ylabel = kwargs.pop('ylabel', 'time [sec]')
        ymax = kwargs.get('ymax')
        ymin = kwargs.get('ymin')

        groups = dict(kwargs.pop('groups'))
        groups, gvals = zip(*groups.items()) if groups else ([], [])
        params = dict(kwargs.pop('params'))
        pvals = zip(*sorted(params))[1]

        nregions = len(regions)
        ngroups = int(np.prod([len(g) for g in gvals]))
        speedup_group = speedup and len(speedup) <= len(gvals)
        speedup_single = speedup and len(speedup) == len(gvals) + 1
        if speedup_group:
            gvals = list(gvals)
            for i, s in enumerate(speedup):
                gvals[i] = filter(lambda x: x != s, gvals[i])
        if speedup_single:
            if speedup[0] in xvals:
                xvals = [i for i in xvals if i not in speedup]
        offset = np.arange(len(xvals)) + 0.1
        if colors:
            colors = colors[:max(nregions, ngroups)]
        else:
            # Set the default color cycle according to the given color map
            cmap = mpl.cm.get_cmap(name=colormap)
            # Colour by region or group, whichever there are more of
            colors = [cmap(i) for i in np.linspace(0, 0.9, max(nregions, ngroups))]
        ax.set_color_cycle(colors)
        linestyles = ('solid', 'dashed', 'dashdot', 'dotted')
        fillstyles = ('', '/', '\\', '-')

        def group(r):
            for i, g in enumerate(bargroups):
                if g in r:
                    return i
            return 0

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
        if kind in ['bar', 'barlog']:
            w = (2*len(speedup) if speedup_group else 1) * 0.8 / (nregions * ngroups)
        else:
            w = 0.8 / len(bargroups)
        i = 0
        for g, gv in enumerate(product(*gvals)):
            for ir, r in enumerate(regions):
                params.update(zip(groups, gv))
                if baseline and baseline[0] in gv:
                    yvals = np.array([self.lookup(r, params, [(xaxis, baseline[1])]) for _ in xvals])
                else:
                    yvals = np.array([self.lookup(r, params, [(xaxis, v)]) for v in xvals])
                if np.isnan(yvals).all():
                    continue
                # Skip parameters used for speedup when generating label
                skip = len(speedup) if speedup_group else 0
                rlabel = [] if nregions == 1 else [r]
                if labels == 'compact':
                    label = ', '.join(rlabel + map(str, gv[skip:]))
                elif labels == 'long':
                    label = ', '.join(rlabel + ['%s: %s' % _ for _ in zip(groups[skip:], gv[skip:])])
                elif isinstance(labels, dict):
                    label = labels[gv]
                # 1) speedup relative to a specimen in the group
                if speedup_group:
                    params.update(zip(groups, speedup))
                    yvals = np.array([self.lookup(r, params, [(xaxis, v)]) for v in xvals]) / yvals
                # 2) speedup relative to a single datapoint
                elif speedup_single:
                    params.update(zip(groups, speedup[1:]))
                    yvals = self.lookup(r, params, [(xaxis, speedup[0])]) / yvals
                if transform:
                    yvals = transform(xvals, yvals)
                if kind in ['barstacked', 'barstackedlog']:
                    plot(offset + group(r) * w, yvals, w,
                         bottom=ystack[group(r)], label=label,
                         color=colors[ir], hatch=fillstyles[g % 4],
                         log=kind == 'barstackedlog')
                    ystack[group(r)] += yvals
                elif kind in ['bar', 'barlog']:
                    plot(offset + i * w, yvals, w, label=label,
                         color=colors[ir], hatch=fillstyles[g % 4],
                         log=kind == 'barlog')
                else:
                    if baseline and baseline[0] in gv:
                        plot(xvalues, yvals, label=label, lw=linewidth, color='k')
                    else:
                        linestyle = linestyles[(g if nregions >= ngroups else ir) % 4]
                        if trendline:
                            plot(xvalues, xvalues[0]*yvals[0]/xvalues,
                                 lw=1, color='k', linestyle=linestyle,
                                 label=trendline)
                            # prevent creating multiple legend entried
                            # (labels starting with _ are ignored)
                            trendline = '_'
                        plot(xvalues, yvals, label=label, lw=linewidth, markeredgecolor='none',
                             linestyle=linestyle, **plotstyle.get(r, {}))
                i += 1
        # Plot custom lines
        for yvals, kargs in lines:
            if np.isscalar(yvals):
                yvals = [yvals] * len(xvalues)
            plot(xvalues, yvals, **kargs)
        # Scale current axis horizontally
        if wscale:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * wscale, box.height])
        # Scale current axis vertically
        if hscale:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width, box.height * hscale])
        if legend is not False:
            l = ax.legend(prop=fontP, framealpha=.5, handlelength=4, **legend)
            l.get_frame().set_color('white')
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title == self.name:
            tsuff = ', '.join('%s=%s' % (k, v) for k, v in params)
            ax.set_title(title + ': ' + tsuff)
        elif title:
            ax.set_title(title % dict(params))
        if grid:
            ax.grid()
        if kind in ['barstacked', 'barstackedlog', 'bar', 'barlog']:
            ax.set_xticks(offset + i * w / 2)
        elif xvalues:
            ax.set_xticks(xvalues)
        if xticklabels:
            ax.set_xticklabels(xticklabels)
        if xtickbins and kind == 'plot':
            ax.locator_params(axis='x', nbins=xtickbins)
        if axis == 'tight':
            ax.axis('tight')
            x0, x1, y0, y1 = ax.axis()
            ax.axis([x0 * .9, x1 * 1.1, y0 * 0.9, y1 * 1.1])
        ax.set_xlim(left=xmin, right=xmax)
        ax.set_ylim(bottom=ymin, top=ymax)
        if hidexticks:
            xticks = ax.xaxis.get_major_ticks()
            for i in hidexticks:
                xticks[i].label.set_visible(False)
        if hideyticks:
            yticks = ax.yaxis.get_major_ticks()
            for i in hideyticks:
                yticks[i].label.set_visible(False)
        if ticksize:
            ax.tick_params(axis='both', which='major',
                           labelsize=ticksize)

    def plot(self, xaxis, **kwargs):
        """Plot results.

        :param xaxis: the parameter to plot on the x-axis
        :param kwargs: keyword arguments override values given in the results
            * figname: base name of output file
            * figsize: figure size (defaults to (9, 6))
            * params: benchmark parameters
            * groups: list of parameters to group (group parameters will be
                shown in the same plot rather than creating multiple plots)
            * legend: dictionary of legend options, passed as keyword argument
                to the matplotlib legend function or False to suppress legend
            * format: comma-separated list of output formats (defaults to svg)
            * hspace: height of space between subplots
            * html: generate an html overview page for plots
            * plotdir: output directory
            * kinds: comma-separated list of kinds of plots:
                * bar: bar plot
                * barstacked: stacked bar plot
                * barlog: bar plot with logarithmic y-axis
                * barstackedlog: stacked bar plot with logarithmic y-axis
                * plot: regular plot
                * semilogx: plot with logarithmic x-axis
                * semilogy: plot with logarithmic y-axis
                * loglog: log-log plot
            * sharex: share x axis (only for subplots)
            * sharey: share y axis (only for subplots)
            * subplot: instead of creating multiple plots, put them side by
                side as subplots with a shared y-axis (defaults to `False`)
            * subplots: create multiple subplots (2-tuple with number of rows
                and columns for the subplots)
            * subplotargs: subplot arguments (used in combination with
                subplots), dictionary with an entry for each row/column given
                by subplots. Yes are row, column and values are dictionaries
                of parameters overriding keyword arguments for each individual
                subplot
            * wspace: width of space between subplots
            * xlabel: x-axis label
            * xvals: values to use for x axis (overrides parameters)
        """
        if rank > 0:
            return
        figname = kwargs.pop('figname', self.result['name'])
        figsize = kwargs.pop('figsize', (9, 6))
        params = dict(kwargs.pop('params', self.result['params']))
        groups = kwargs.get('groups', [])
        legend = kwargs.get('legend', {'loc': 'best'})
        format = kwargs.pop('format', 'svg')
        hspace = kwargs.get('hspace')
        html = kwargs.get('html')
        plotdir = kwargs.pop('plotdir', self.plotdir)
        kinds = kwargs.pop('kinds', 'plot')
        sharex = kwargs.get('sharex', 'none')
        sharey = kwargs.get('sharey', 'none')
        speedup = kwargs.get('speedup')
        subplot = kwargs.get('subplot')
        subplots = kwargs.get('subplots')
        subplotargs = kwargs.get('subplotargs')
        wspace = kwargs.get('wspace')
        if subplots:
            xlabel = kwargs.pop('xlabel', None)
            title = kwargs.pop('title', None)
        if not path.exists(plotdir):
            makedirs(plotdir)

        kwargs['xvals'] = kwargs.pop('xvals', params.pop(xaxis))
        kwargs['groups'] = zip(groups, [params.pop(g) for g in groups])
        pkeys, pvals = zip(*sorted(params.items()))

        nv = len(list(product(*pvals)))

        def save(fig, fname, outline, extra_artists=[]):
            if not format:
                fig.show()
            else:
                for fmt in format.split(','):
                    fname += '.' + fmt
                    fig.savefig(path.join(plotdir, fname),
                                orientation='landscape', format=fmt,
                                transparent=True, bbox_inches='tight',
                                bbox_extra_artists=extra_artists)
                    if fmt in ['svg', 'png']:
                        outline += ['<td><img src="%s"></td>' % fname]
            plt.close(fig)

        for kind in kinds.split(','):
            outline = []
            if subplot:
                axes = []
                fig = plt.figure(figname + '_' + kind, figsize=figsize, dpi=300)
            for p, pv in enumerate(product(*pvals), 1):
                pdict = zip(pkeys, pv)
                fsuff = '_'.join('%s%s' % (k, str(v).replace('.', '_')) for k, v in pdict)
                # Append speedup to file base name if any
                if speedup:
                    fsuff += '_speedup' + ''.join(speedup)
                if subplots:
                    nrows, ncols = subplots
                    fig, ax = plt.subplots(nrows, ncols, sharex, sharey,
                                           num=figname + '_' + fsuff,
                                           squeeze=False,
                                           figsize=figsize, dpi=300)
                    for r in range(nrows):
                        for c in range(ncols):
                            kargs = copy(kwargs)
                            kargs['title'] = None
                            kargs['axis'] = 'tight'
                            kargs.update(subplotargs[r, c])
                            self.subplot(ax[r][c], xaxis, kind, params=pdict, **kargs)
                    # Adjust space between subplots
                    fig.subplots_adjust(hspace=hspace, wspace=wspace)
                    if title:
                        fig.suptitle(title)
                    if legend and legend != {'loc': 'best'}:
                        lhandles, llabels = ax[r][c].get_legend_handles_labels()
                        l = fig.legend(lhandles, llabels, prop=fontP,
                                       framealpha=.5, handlelength=4, **legend)
                        l.get_frame().set_color('white')
                    extra_artists = []
                    if xlabel:
                        extra_artists.append(fig.text(0.5, -0.03, xlabel, ha='center'))
                    outline += ['<tr>']
                    save(fig, '%s_%s_%s' % (figname, kind, fsuff), outline, extra_artists)
                    outline += ['</tr>']
                elif subplot:
                    ax = fig.add_subplot(1, nv, p, sharey=(axes[p-2] if p > 1 else None))
                    axes.append(ax)
                    kargs = copy(kwargs)
                    kargs['legend'] = False
                    kargs['axis'] = 'tight'
                    if p > 1:
                        kargs['ylabel'] = None
                    if subplotargs:
                        kargs.update(subplotargs[p])
                    self.subplot(ax, xaxis, kind, params=pdict, **kargs)
                else:
                    fig = plt.figure(figname + '_' + fsuff, figsize=figsize, dpi=300)
                    ax = fig.add_subplot(111)
                    self.subplot(ax, xaxis, kind, params=pdict, **kwargs)
                    outline += ['<tr>']
                    save(fig, '%s_%s_%s' % (figname, kind, fsuff), outline)
                    outline += ['</tr>']
            if subplot:
                # Remove space between subplots
                fig.subplots_adjust(hspace=hspace, wspace=wspace)
                # Hide y ticks for all but left plot
                plt.setp([a.get_yticklabels() for a in fig.axes[1:]], visible=False)
                lhandles, llabels = ax.get_legend_handles_labels()
                l = fig.legend(lhandles, llabels, prop=fontP, framealpha=.5,
                               handlelength=4, **legend)
                l.get_frame().set_color('white')
                outline += ['<tr>']
                save(fig, '%s_%s' % (figname, kind), outline)
                outline += ['</tr>']
            if html:
                fname = '%s_%s_%s_%s.html' % (figname, xaxis, '_'.join(groups), kind)
                with open(path.join(plotdir, fname), 'w') as f:
                    f.write('\n'.join(outline))

    def archive(self, dirname=None):
        """Archive results, profiles and plots in a timestamped directory."""
        timestamp = datetime.now().strftime('%Y-%m-%dT%H%M%S')
        dirname = dirname or path.join(self.basedir, timestamp)
        makedirs(dirname)
        for d in [self.resultsdir, self.profiledir, self.plotdir]:
            shutil.move(d, dirname)
