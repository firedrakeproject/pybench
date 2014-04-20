from time import sleep

from pybench import Benchmark


def test_no_params():
    assert Benchmark().run(method=lambda: None)['timings'][()]['total'] > 0.0


def test_sleep():
    def myfunc(n, duration):
        for _ in range(n):
            sleep(duration)
    times = Benchmark().run(method=myfunc,
                            params={'n': range(3), 'duration': (0.001, 0.002)})
    for (n, d), t in times['timings'].items():
        assert abs(n*d - t['total']) < 1e-3
