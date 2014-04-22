from time import sleep

from pybench import Benchmark


class TimedRegion(Benchmark):
    def test(self):
        with self.timed_region('stuff'):
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
                            params={'n': range(3), 'duration': (0.001, 0.002)})
    for (n, d), t in times['timings'].items():
        assert abs(n*d - t['total']) < 1e-3


def test_timed_region():
    result = TimedRegion().run()
    assert result['timings']['total'] > 0.0
    assert result['timings']['stuff'] > 0.0


def test_save_load(tmpdir):
    b = TimedRegion()
    d = tmpdir.join(b.name).strpath
    result = b.run()
    b.save(d)
    TimedRegion().load(d) == result
