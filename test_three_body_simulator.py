import pytest
import numpy as np
from sklearn.neighbors import KDTree
import json

from three_body_common import SimulationParams

# Import the functions to test
from three_body_simulator import (
    modified_three_body,
    estimate_epsilon_range,
    check_connectedness,
    box_counting_dim,
    random_initial_conditions,
    run_simulation,
    main,
)


def test_modified_three_body():
    state = np.array([1, 1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0])
    epsilon = 0.5
    result = modified_three_body(0, state, epsilon)

    assert len(result) == 12
    assert isinstance(result, list)

    # Check if the accelerations are correct
    grav = 1  # gravitational constant
    expected_r = np.sqrt(2**2 + (0.5 * 2) ** 2)  # distance between bodies 1 and 2
    expected_ax1 = grav * (-2) / expected_r**3  # acceleration of body 1 in x direction
    expected_ay1 = (
        epsilon * grav * (-2) / expected_r**3
    )  # acceleration of body 1 in y direction

    assert np.isclose(result[2], expected_ax1, rtol=1e-7)
    assert np.isclose(result[3], expected_ay1, rtol=1e-7)

    # Check symmetry
    assert np.isclose(result[2], -result[6], rtol=1e-7)  # ax1 should be opposite to ax2
    assert np.isclose(result[3], -result[7], rtol=1e-7)  # ay1 should be opposite to ay2

    # Check that body 3 (at origin) experiences balanced forces
    assert np.isclose(result[10], 0, atol=1e-7)  # ax3 should be close to 0
    assert np.isclose(result[11], 0, atol=1e-7)  # ay3 should be close to 0


def test_estimate_epsilon_range():
    points = np.array([[0, 0], [1, 1], [2, 2]])
    min_eps, max_eps = estimate_epsilon_range(points)
    assert min_eps < max_eps
    assert min_eps > 0


@pytest.mark.parametrize("epsilon,expected", [(1.5, True), (0.5, False)])
def test_check_connectedness(epsilon, expected):
    points = np.array([[0, 0], [1, 1], [2, 2]])
    tree = KDTree(points)
    assert check_connectedness(tree, epsilon) == expected


def test_box_counting_dim():
    np.random.seed(42)  # for reproducibility
    points_2d = np.random.rand(
        1000, 2
    )  # 2D random points should have dimension close to 2
    dim, log_eps, log_num_points = box_counting_dim(points_2d)
    assert 1.8 < dim < 2.2  # allowing some tolerance
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
