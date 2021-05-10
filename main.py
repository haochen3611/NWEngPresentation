import numba

from numba import jit
import random


@jit(nopython=True)
def monte_carlo_pi(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples


def monte_carlo_pi_nonjit(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples


if __name__ == '__main__':
    import timeit
    n_points = int(1e6)
    # print(monte_carlo_pi(n_points))
    time_jit = timeit.timeit(lambda: monte_carlo_pi(n_points), number=10)
    print(f"Time with JIT: {time_jit} sec\n")
    time_nonjit = timeit.timeit(lambda: monte_carlo_pi_nonjit(n_points), number=10)
    print(f"Time without JIT: {time_nonjit} sec\n"
          f"Speed up: {time_nonjit/time_jit: .1f}x")
