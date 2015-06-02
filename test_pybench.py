import pytest

from pybench import Benchmark


class TestBenchmark(Benchmark):
    params = {'a': range(1, 4), 'b': range(2, 7, 2)}
    regions = ['total', 'r1', 'r2']

    def test(self, a=None, b=None):
        with self.timed_region('r1'):
            pass
        with self.timed_region('r2'):
            pass


class TimedRegion(Benchmark):
    regions = ['total', 'stuff']

    def test(self):
        with self.timed_region('stuff'):
            pass


class InvalidRegion(Benchmark):
    def test(self):
        with self.timed_region('stuff'):
            pass


class Parametrized(Benchmark):
    params = {'a': range(3), 'b': range(3)}

    def test(self, a=None, b=None):
        pass


def test_no_params():
    assert Benchmark().run(method=lambda: None)(region='total') > 0.0


def test_no_repeats():
    assert all(Benchmark().run(method=lambda: None, repeats=0).data.isnull())


def test_invalid_region():
    with pytest.raises(ValueError):
        InvalidRegion().run()


def test_invalid_param():
    with pytest.raises(ValueError):
        Parametrized().run(params={'c': [1]})


def test_invalid_param_value():
    with pytest.raises(ValueError):
        Parametrized().run(params={'a': [3]})


def test_timed_region():
    bench = TimedRegion().run()
    assert bench(region='total') > 0.0
    assert bench(region='stuff') > 0.0


def test_save_load(tmpdir):
    b = TimedRegion()
    d = tmpdir.join(b.name).strpath
    result = b.run()
    b.save(d)
    TimedRegion().load(d) == result


def test_combine_arrays(tmpdir):
    a = TestBenchmark().run()
    b = TestBenchmark().run()
    c = TestBenchmark().combine('c', ['a', 'b'], [a.data, b.data])
    assert 'c' in c.data.coords
    assert all(c.data.coords['c'] == ['a', 'b'])
    assert (c(c='a') == a.data).all()
    assert (c(c='b') == b.data).all()


def test_parametrized():
    assert (Parametrized().run().data > 0.0).all()


def test_combine_parametrized(tmpdir):
    da = tmpdir.join('a').strpath
    db = tmpdir.join('b').strpath
    b = Parametrized().run().save(da).save(db)
    c = Parametrized().combine('c', ['a', 'b'], [da, db])
    assert 'c' in c.data.coords
    assert all(c.data.coords['c'] == ['a', 'b'])
    assert (c(c='a') == b.data).all()
    assert (c(c='b') == b.data).all()
