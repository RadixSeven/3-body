import pytest
import numpy as np
from pytest import approx
from sklearn.neighbors import KDTree
import json

from three_body_common import SimulationParams

# Import the functions to test
from three_body_simulator import (
    modified_three_body,
    estimate_epsilon_range,
    is_connected,
    box_counting_dim,
    random_initial_conditions,
    run_simulation,
    main,
)


def test_modified_three_body():
    # Three stationary points: one at 1,1, one at -1,-1 and one at 0,0.
    state = np.array([1, 1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0])
    y_grav = 0.5
    result = modified_three_body(0, state, y_grav)
    # Actual values
    vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2, vx3, vy3, ax3, ay3 = result

    assert len(result) == 12

    # Check if the accelerations are correct
    expected_r12 = np.sqrt(2**2 + (2**2))  # distance between bodies 1 and 2
    expected_fx12 = -2 / (expected_r12**3)  # force on body 1 from body 2 in x dir
    expected_fy12 = -2 * y_grav / (expected_r12**3)  # force on body 1 from 2 in y dir
    expected_r13 = np.sqrt(1**2 + 1**2)  # distance between bodies 1 and 3
    expected_fx13 = -1 / (expected_r13**3)  # force on body 1 from body 3 in x dir
    expected_fy13 = -1 * y_grav / (expected_r13**3)
    expected_ax1 = expected_fx12 + expected_fx13
    expected_ay1 = expected_fy12 + expected_fy13

    assert ax1 == approx(expected_ax1, rel=1e-7)
    assert ay1 == approx(expected_ay1, rel=1e-7)

    # Check symmetry
    assert ax1 == approx(-ax2, rel=1e-7)  # ax1 should be opposite to ax2
    assert ay1 == approx(-ay2, rel=1e-7)  # ay1 should be opposite to ay2

    # Check that body 3 (at origin) experiences balanced forces
    assert ax3 == approx(0, abs=1e-7)  # ax3 should be close to 0
    assert ay3 == approx(0, abs=1e-7)  # ay3 should be close to 0


def test_estimate_epsilon_range():
    points = np.array([[0, 0], [1, 1], [2, 2]])
    min_eps, max_eps = estimate_epsilon_range(points)
    assert min_eps < max_eps
    assert min_eps > 0


@pytest.mark.parametrize("epsilon,expected", [(1.5, True), (0.5, False)])
def test_check_connectedness(epsilon, expected):
    points = np.array([[0, 0], [1, 1], [2, 2]])
    tree = KDTree(points)
    assert is_connected(tree, epsilon) == expected


def test_box_counting_dim():
    np.random.seed(42)  # for reproducibility
    points_2d = np.random.rand(
        1000, 2
    )  # 2D random points should have dimension close to 2
    dim, log_eps, log_num_points = box_counting_dim(points_2d)
    assert dim == approx(2, abs=0.2)  # allowing some tolerance
    assert len(log_eps) == len(log_num_points)


def test_random_initial_conditions():
    init_cond = random_initial_conditions()
    assert len(init_cond) == 12
    reshaped = init_cond.reshape(3, 4)
    assert np.allclose(np.mean(reshaped[:, :2]), 0, atol=1e-10)


def test_run_simulation():
    epsilon = 0.5
    t_span = (0, 10)
    num_points = 1000
    dim, log_eps, log_num_points = run_simulation(epsilon, t_span, num_points)
    assert isinstance(dim, float)
    assert len(log_eps) == len(log_num_points)


def test_main_function(tmp_path):
    # Using pytest's tmp_path fixture for temporary directory
    output_file = tmp_path / "output.json"

    p = SimulationParams(
        epsilon=[0, 0.5], trials=2, time=10, points=1000, output=str(output_file)
    )
    main(p)

    # Check if the file was created and contains valid JSON
    assert output_file.exists()
    with open(output_file, "r") as f:
        data = json.load(f)

    assert "0" in data
    assert "0.5" in data
    assert len(data["0"]) == 2
    assert len(data["0.5"]) == 2


# Optional: Add a fixture for common test data
@pytest.fixture
def sample_points():
    return np.array([[0, 0], [1, 1], [2, 2]])


# Example of using the fixture
def test_with_fixture(sample_points):
    min_eps, max_eps = estimate_epsilon_range(sample_points)
    assert min_eps < max_eps


if __name__ == "__main__":
    pytest.main()
