"""Property tests for the from-scratch TSP solver (greedy NN + 2-opt)."""

import numpy as np
import pytest

from inksep.tsp import nearest_neighbour, open_path_length, solve_tsp, two_opt


def _grid_points(n: int = 60, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 100, size=(n, 2))


def test_nn_visits_every_point_once():
    pts = _grid_points(80)
    order = nearest_neighbour(pts)
    assert sorted(order.tolist()) == list(range(80))


def test_nn_edge_cases():
    empty = nearest_neighbour(np.empty((0, 2)))
    assert len(empty) == 0
    single = nearest_neighbour(np.array([[1.0, 2.0]]))
    assert single.tolist() == [0]
    pair = nearest_neighbour(np.array([[0.0, 0.0], [3.0, 4.0]]))
    assert set(pair.tolist()) == {0, 1}


def test_nn_rejects_out_of_range_start():
    pts = _grid_points(5)
    with pytest.raises(ValueError, match="start"):
        nearest_neighbour(pts, start=5)
    with pytest.raises(ValueError, match="start"):
        nearest_neighbour(pts, start=-1)


def test_nn_deterministic():
    pts = _grid_points(50, seed=3)
    assert nearest_neighbour(pts).tolist() == nearest_neighbour(pts).tolist()


def test_two_opt_shortens_crossing_case():
    """A deliberately bad zig-zag order must be improved by 2-opt."""
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    bad = np.array([0, 2, 1, 3], dtype=np.intp)  # crosses the square diagonally
    before = open_path_length(pts, bad)
    after_order = two_opt(pts, bad, budget_s=None)
    after = open_path_length(pts, after_order)
    assert after < before
    assert sorted(after_order.tolist()) == [0, 1, 2, 3]


def test_two_opt_never_worsens():
    pts = _grid_points(40, seed=7)
    start_order = nearest_neighbour(pts)
    before = open_path_length(pts, start_order)
    result = two_opt(pts, start_order, budget_s=None)
    assert open_path_length(pts, result) <= before + 1e-9
    assert sorted(result.tolist()) == list(range(40))


def test_two_opt_does_not_mutate_input():
    pts = _grid_points(20, seed=1)
    order = nearest_neighbour(pts)
    snapshot = order.copy()
    two_opt(pts, order, budget_s=None)
    assert order.tolist() == snapshot.tolist()


def test_two_opt_small_n_is_noop():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    order = np.array([2, 0, 1], dtype=np.intp)
    result = two_opt(pts, order, budget_s=None)
    assert result.tolist() == order.tolist()


def test_two_opt_budget_truncates_but_valid():
    pts = _grid_points(200, seed=11)
    order = nearest_neighbour(pts)
    before = open_path_length(pts, order)
    # Zero budget: zero passes, still a valid never-worse permutation
    result = two_opt(pts, order, budget_s=0.0)
    assert sorted(result.tolist()) == list(range(200))
    assert open_path_length(pts, result) <= before + 1e-9


def test_two_opt_respects_max_passes():
    pts = _grid_points(100, seed=13)
    order = nearest_neighbour(pts)
    r0 = two_opt(pts, order, budget_s=None, max_passes=0)
    r1 = two_opt(pts, order, budget_s=None, max_passes=1)
    assert r0.tolist() == order.tolist()  # no passes applied
    assert open_path_length(pts, r1) <= open_path_length(pts, order) + 1e-9


def test_solve_tsp_is_permutation_and_never_worse():
    pts = _grid_points(120, seed=17)
    naive = np.arange(120, dtype=np.intp)
    naive_len = open_path_length(pts, naive)
    result = solve_tsp(pts, budget_s=None)
    assert sorted(result.tolist()) == list(range(120))
    assert open_path_length(pts, result) <= naive_len


def test_solve_tsp_beats_random_order_statistically():
    """On a shuffled point set, solve_tsp must shorten the path meaningfully."""
    pts = _grid_points(150, seed=19)
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(150).astype(np.intp)
    before = open_path_length(pts, shuffled)
    after = open_path_length(pts, solve_tsp(pts, budget_s=None))
    assert after < before * 0.75  # at least 25% shorter on random order


def test_solve_tsp_edge_cases():
    assert solve_tsp(np.empty((0, 2))).tolist() == []
    assert solve_tsp(np.array([[5.0, 5.0]])).tolist() == [0]


def test_solve_tsp_deterministic_with_fixed_passes():
    pts = _grid_points(80, seed=23)
    a = solve_tsp(pts, budget_s=None, max_passes=5)
    b = solve_tsp(pts, budget_s=None, max_passes=5)
    assert a.tolist() == b.tolist()


def test_open_path_length_open_not_closed():
    """Open path: no closing edge back to the start."""
    pts = np.array([[0.0, 0.0], [3.0, 4.0]])
    order = np.array([0, 1], dtype=np.intp)
    # Open length is 5.0; a closed tour would be 10.0
    assert open_path_length(pts, order) == pytest.approx(5.0)


def test_open_path_length_single_point():
    assert open_path_length(np.array([[1.0, 1.0]]), np.array([0], dtype=np.intp)) == 0.0
