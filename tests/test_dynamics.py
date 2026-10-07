"""Fonctions mathématiques pures : exactitude et indépendance au pas de temps."""
import math
import random

from valdar.heart.dynamics import approach_one, in_window, ou_noise, relax, soft_add


def test_relax_is_exact_and_step_independent():
    one = relax(1.0, 0.2, 600.0, 300.0)
    many = 1.0
    for _ in range(600):
        many = relax(many, 0.2, 1.0, 300.0)
    assert math.isclose(one, many, rel_tol=1e-9)
    assert math.isclose(one, 0.2 + 0.8 * math.exp(-2.0), rel_tol=1e-12)


def test_relax_keeps_most_of_a_slow_deviation_after_one_second():
    # régression du bug OpenCode : un écart de constante 2 h ne doit pas disparaître en 1 s
    x = relax(1.0, 0.15, 1.0, 7200.0)
    assert x > 0.99


def test_soft_add_never_crosses_bounds_and_is_linear_near_center():
    assert soft_add(0.5, 0.01) == __import__("pytest").approx(0.51, abs=2e-4)
    x = 0.5
    for _ in range(100):
        x = soft_add(x, 0.5)
    assert 0.0 <= x <= 1.0
    y = 0.5
    for _ in range(100):
        y = soft_add(y, -0.5)
    assert 0.0 <= y <= 1.0


def test_ou_noise_variance_does_not_depend_on_step():
    def stationary_std(dt: float, n: int) -> float:
        rng = random.Random(1)
        x, xs = 0.0, []
        for _ in range(n):
            x = x * math.exp(-dt / 100.0) + ou_noise(rng, dt, 100.0, 0.05)
            xs.append(x)
        xs = xs[n // 5:]
        m = sum(xs) / len(xs)
        return math.sqrt(sum((v - m) ** 2 for v in xs) / len(xs))

    assert abs(stationary_std(1.0, 200_000) - 0.05) < 0.006
    assert abs(stationary_std(30.0, 20_000) - 0.05) < 0.006


def test_approach_one_composes():
    a = approach_one(0.0, 7200.0, 3600.0)
    b = approach_one(approach_one(0.0, 3600.0, 3600.0), 3600.0, 3600.0)
    assert math.isclose(a, b, rel_tol=1e-12)


def test_in_window_handles_midnight():
    assert in_window(23.0, "22:30", "09:00")
    assert in_window(3.0, "22:30", "09:00")
    assert not in_window(12.0, "22:30", "09:00")
    assert in_window(12.0, "07:00", "23:30")
