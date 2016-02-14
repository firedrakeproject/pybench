from itertools import product
from time import sleep

from pybench import Benchmark


class TimedRegion(Benchmark):
    def test(self):
        with self.timed_region('stuff'):
            pass


class Parametrized(Benchmark):
    params = [('a', range(3)), ('b', range(3))]

    def test(self, a=None, b=None):
        pass


def test_no_params():
    assert Benchmark().run(method=lambda: None)['timings']['total'] > 0.0


def test_no_repeats():
    assert not Benchmark().run(method=lambda: None, repeats=0)['timings']


def test_sleep():
    def myfunc(n, duration):
        for _ in range(n):
            sleep(duration)
    times = Benchmark().run(method=myfunc,
                            params=[('n', range(3)),
                                    ('duration', (0.001, 0.002))])
    for (n, d), t in times['timings'].items():
        assert abs(n*d - t['total']) < 1e-3


def test_timed_region():
    result = TimedRegion().run()
    assert result['timings']['total'] >= 0.0
    assert result['timings']['stuff'] >= 0.0


def test_save_load(tmpdir):
    b = TimedRegion()
    d = tmpdir.join(b.name).strpath
    result = b.run()
    b.save(d)
    TimedRegion().load(d) == result


def test_combine_regions(tmpdir):
    b = TimedRegion()
    da = tmpdir.join('a').strpath
    db = tmpdir.join('b').strpath
    keys = b.run()['timings'].keys()
    b.save(da)
    b.save(db)
    result = TimedRegion().combine({da: 'a', db: 'b'})
    assert all('a ' + k in result['timings'] for k in keys)
    assert all('b ' + k in result['timings'] for k in keys)


def test_parametrized():
    result = Parametrized().run()
    _, pvalues = zip(*result['params'])
    assert all(result['timings'][p]['total'] >= 0.0
               for p in product(*pvalues))


def test_combine_parametrized(tmpdir):
    b = Parametrized()
    da = tmpdir.join('a').strpath
    db = tmpdir.join('b').strpath
    params = b.run()['timings'].keys()
    b.save(da)
    b.save(db)
    result = Parametrized().combine({da: 'a', db: 'b'})
    assert all('a total' in result['timings'][p] for p in params)
    assert all('b total' in result['timings'][p] for p in params)
