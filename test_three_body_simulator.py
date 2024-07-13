from io import StringIO
from unittest.mock import patch

import pytest
import numpy as np
from pytest import approx
from sklearn.neighbors import KDTree
import json

from three_body_common import SimulationParams, SimulationResult

# Import the functions to test
from three_body_simulator import (
    modified_three_body,
    estimate_epsilon_range,
    is_connected,
    box_counting_dim,
    random_initial_conditions,
    run_simulation,
    main,
    all_offsets,
    are_8_connected,
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


def test_box_counting_dim_400_grid():
    points_2d = np.array([[i, j] for i in range(20) for j in range(20)])
    dim, log_eps, log_num_points = box_counting_dim(points_2d)
    assert dim == approx(2, abs=0.1)  # allowing some tolerance
    assert len(log_eps) == len(log_num_points)


def test_box_counting_dim_900_grid():
    n = 30
    points_2d = np.array([[i, j] for i in range(n) for j in range(n)])
    dim, log_eps, log_num_points = box_counting_dim(points_2d)
    assert dim == approx(2, abs=0.1)  # allowing some tolerance
    assert len(log_eps) == len(log_num_points)


def test_box_counting_dim_random_500():
    np.random.seed(42)
    points_2d = np.random.rand(500, 2)
    dim, log_eps, log_num_points = box_counting_dim(points_2d)
    assert dim == approx(2, abs=0.2)  # allowing some tolerance
    assert len(log_eps) == len(log_num_points)


def test_random_initial_conditions():
    np.random.seed(42)
    init_cond = random_initial_conditions()
    assert len(init_cond) == 12
    reshaped = init_cond.reshape(3, 4)
    assert np.allclose(np.mean(reshaped[:, :2]), 0, atol=1e-10)


def test_are_8_connected():
    assert are_8_connected({(0, 0), (0, 1)})
    assert not are_8_connected({(0, 0), (0, 2)})
    assert are_8_connected({(0, 0), (1, 1)})
    assert are_8_connected({(0, 0), (1, 1), (0, 2)})
    assert are_8_connected(
        {
            (0, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0),
            (1, 1, 0, 0, 0, 0),
            (1, 2, 1, 0, 0, 0),
            (1, 3, 1, 1, 0, 0),
        }
    )
    assert not are_8_connected(
        {
            (0, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0),
            (2, 1, 0, 0, 0, 0),
            (3, 2, 1, 0, 0, 0),
            (4, 3, 2, 2, 0, 0),
        }
    )


def test_run_simulation():
    result = run_simulation(epsilon=1, t_span=(0, 10), num_points=1000)
    assert len(result.log_eps) == len(result.log_N)
    assert result.dimension > 0
    assert result.lyapunov > 0  # epsilon=1 should be chaotic
    result = run_simulation(epsilon=0, t_span=(0, 10), num_points=1000)
    assert result.dimension > 0


def test_all_offsets():
    with pytest.raises(ValueError):
        all_offsets(0)
    assert sorted(map(tuple, all_offsets(1))) == sorted({(1,), (-1,)})
    assert sorted(map(tuple, all_offsets(2))) == sorted(
        {(0, 1), (0, -1), (1, 1), (1, 0), (1, -1), (-1, 1), (-1, 0), (-1, -1)}
    )


class NonClosingStringIO(StringIO):
    def close(self):
        # Override close to do nothing
        pass


@patch("three_body_simulator.run_simulation")
@patch("builtins.open")
def test_main(mock_open, mock_run_simulation):
    # Mock data for initial file read
    mock_initial_data = {
        "0": [
            {
                "initial_conditions": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1],
                "dimension": 2,
                "log_eps": [0, 1, 2],
                "log_N": [0, 2, 4],
                "lyapunov": 0.5,
            }
        ]
    }
    mock_initial_file = StringIO(json.dumps(mock_initial_data))
    mock_initial_file.name = str("output.json")

    # Mock data for run_simulation
    mock_simulation_results = [
        # Epsilon = 0
        SimulationResult(
            initial_conditions=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            dimension=2,
            log_eps=[0, 1, 2],
            log_N=[0, 2, 4],
            lyapunov=0.5,
        ),
        # Epsilon = 0
        SimulationResult(
            initial_conditions=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 22],
            dimension=2,
            log_eps=[0, 1, 2],
            log_N=[0, 2, 4],
            lyapunov=0.5,
        ),
        # Epsilon = 1
        SimulationResult(
            initial_conditions=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            dimension=3,
            log_eps=[0, 1, 2],
            log_N=[0, 3, 6],
            lyapunov=1.6,
        ),
        # Epsilon = 1
        SimulationResult(
            initial_conditions=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 24],
            dimension=3,
            log_eps=[0, 1, 2],
            log_N=[0, 3, 6],
            lyapunov=1.6,
        ),
    ]

    # Mock for file writing
    mock_output = NonClosingStringIO()

    # Set up the mock_open to return different file objects for read and write
    mock_open.side_effect = [mock_initial_file, mock_output]

    # Set up mock_run_simulation to return predefined results
    mock_run_simulation.side_effect = mock_simulation_results

    # Create SimulationParams
    p = SimulationParams(
        epsilon=[0, 1], trials=2, time=10, points=1000, output="output.json"
    )

    # Run the main function
    main(p)

    # Check if run_simulation was called the correct number of times
    assert mock_run_simulation.call_count == 4

    # Check the content written to the output file
    mock_output.seek(0)
    written_data = json.loads(mock_output.getvalue())
    written_data = {
        k: [SimulationResult(**d) for d in dicts] for k, dicts in written_data.items()
    }

    assert "0" in written_data
    assert "1" in written_data
    assert len(written_data["0"]) == 3  # 1 initial + 2 new
    assert len(written_data["1"]) == 2  # 0 initial + 2 new

    # Check if the new simulation results were added correctly
    assert written_data["0"][1:] == mock_simulation_results[:2]
    assert written_data["1"] == mock_simulation_results[2:]


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
