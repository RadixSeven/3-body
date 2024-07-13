import logging

import numpy as np
from scipy.integrate import solve_ivp
from sklearn.neighbors import KDTree
import json
from pathlib import Path
import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from tqdm import tqdm

from three_body_common import (
    SimulationResult,
    SimulationParams,
    modified_three_body,
)


def estimate_lyapunov_exponent(
    func: callable,
    y0: np.ndarray,
    t_span: tuple[float, float],
    num_points: int,
    epsilon: float,
    initial_solution: np.ndarray,
    delta: float = 1e-9,
) -> float:
    """
    Estimate the largest Lyapunov exponent of the system.

    Args:
        func (callable): The function defining the ODE system.
        y0 (np.ndarray): Initial condition.
        t_span (tuple[float, float]): Time span for the simulation.
        num_points (int): Number of points to evaluate.
        epsilon (float): Coupling strength for the y-dimension.
        initial_solution (np.ndarray): The solution for the initial condition.
        delta (float): Small perturbation for initial condition. Defaults to 1e-9.

    Returns:
        float: Estimated largest Lyapunov exponent.
    """
    # Use the provided initial solution
    y = initial_solution

    # Perturbed initial condition
    y0_perturbed = y0 + delta * np.random.randn(len(y0))
    sol_perturbed = solve_ivp(
        func,
        t_span,
        y0_perturbed,
        args=(epsilon,),
        dense_output=True,
        t_eval=np.linspace(t_span[0], t_span[1], num_points),
    )
    y_perturbed = sol_perturbed.y

    # Calculate Lyapunov exponent
    d0 = np.linalg.norm(y[:, 0] - y_perturbed[:, 0])
    d1 = np.linalg.norm(y[:, -1] - y_perturbed[:, -1])
    exponent = np.log(d1 / d0) / (t_span[1] - t_span[0])

    return exponent


@dataclass
class BoxCountPair:
    log_eps: float
    log_num_points: float


def box_counting_dim(
    points: np.ndarray, n_samples: int = 20
) -> tuple[float, list[float], list[float]]:
    """
    Estimate the box-counting dimension of a set of points.

    This actually adds spheres around each point rather than
    boxes.

    Args:
        points: Array of points around which boxes will be placed. There must be at least 2.
        n_samples: Number of epsilon values to sample. Defaults to 20.

    Returns:
        Estimated dimension, log(1/epsilon) values, log(num_points) values.
    """
    shape = points.shape
    if shape[0] < 2**2 + 1:
        logging.error(
            f"Need at least 5 points to calculate dimension in a 2-D space. Got: {shape} with contents {points}"
        )
        return np.nan, [], []
    eps_min, eps_max = estimate_epsilon_range(points)
    eps_min = eps_min / np.sqrt(
        2
    )  # We're using squares, not circles, so we can start with smaller radii.
    epsilons = np.logspace(np.log(eps_min), np.log(eps_max), n_samples, base=np.exp(1))

    pairs = []

    for epsilon in epsilons:
        squares = covering_squares(points, epsilon)
        if not are_8_connected_kd(squares):
            continue
        num_squares = len(squares)

        lns = np.log(num_squares)
        if len(pairs) > 0 and lns == pairs[-1].log_num_points:
            # Only record the maximum epsilon that generated
            # a given number of squares unless there are only 4
            # squares, where things start to break down.
            cur_l_eps = pairs[-1].log_eps
            new_l_eps = -np.log(epsilon)
            pairs[-1].log_eps = max(cur_l_eps, new_l_eps)
        else:
            pairs.append(BoxCountPair(-np.log(epsilon), lns))

    pairs = sorted(pairs, key=lambda x: x.log_eps)
    if len(pairs) >= 4:
        pairs = pairs[1:-1]  # Remove the first and last points, which are noisy
    log_eps = [pair.log_eps for pair in pairs]
    log_n = [pair.log_num_points for pair in pairs]

    # Perform linear fit on the connected portion
    if len(log_eps) < 2:
        logging.warning("Not enough box-counting points for dimension estimation.")
        return np.nan, log_eps, log_n

    slope, _ = np.polyfit(log_eps, log_n, 1)
    return slope, log_eps, log_n


def estimate_epsilon_range(points: np.ndarray) -> tuple[float, float]:
    """
    Estimate a suitable range for epsilon values in box-counting dimension estimation based on the data.

    The minimum value is the distance at which all points have at least one neighbor. One could use
    this value/sqrt(2), since that would still result in connected boxes if the points lie along the
    diagonal of a square box. However, we're not using square boxes. We need the radius to another point,
    for which this is too small.

    The maximum value is the maximum distance between any two points in the data. This will result in
    all points being in one bin. This should be OK since we use exactly this epsilon value to calculate
    the fit, so on the log-log plot, it will be a point at [log(max_distance), 0]. The graph flattens out
    after this, which would mess up our fit, but including this point should still be OK.

    Args:
        points: Array of points. Shape: (n_points, n_dimensions).

    Returns:
        Minimum and maximum epsilon values.
    """
    # Distances to the nearest two neighbors comparing the set of points to itself
    pair_distances, _ = KDTree(points).query(points, k=2)
    # distances to nearest neighbors
    # The first nearest neighbor is the point itself, skip it
    distances = pair_distances[:, 1]
    # All points will have a neighbor at max_neighbor_distance
    max_neighbor_distance = np.max(distances)
    # No point will have neighbors at distances greater than max_distance
    max_distance = np.max(np.ptp(points, axis=0))  # Maximum extent of data
    return max_neighbor_distance, max_distance


def _all_offsets_with_0(num_dimensions: int) -> list[list[int]]:
    """
    Same as all_offsets, but includes the [0, 0, ..., 0] offset.
    """
    offsets = [[]]
    for i in range(num_dimensions):
        minus1 = [[*offset, -1] for offset in offsets]
        zero = [[*offset, 0] for offset in offsets]
        plus1 = [[*offset, 1] for offset in offsets]
        offsets = minus1 + zero + plus1
    return offsets


def all_offsets(num_dimensions: int) -> list[list[int]]:
    """
    Generate all possible offsets for a given number of dimensions.

    Args:
        num_dimensions: Number of dimensions. Must be 1 or more.

    Returns:
        List of all possible offsets.
    """
    if num_dimensions == 0:
        raise ValueError(
            "Number of dimensions must be 1 or more when calculating offsets."
        )
    return [offset for offset in _all_offsets_with_0(num_dimensions) if any(offset)]


def are_8_connected(squares: set[tuple]) -> bool:
    """
    Check if the hypercubes identified by the tuples in ``squares`` are connected in a way
    that would be 8-connected in a 2D grid, that is, in a hypercube whose coordinates are one
    away from the connected hypercube.

    Note that the algorithm used here is exponential in the number of dimensions, scaling as O(3^d*len(squares)).

    Args:
        squares: Set of grid-squares, each represented as a tuple of indices.

    Returns:
        True if connected, False otherwise.
    """
    # Create a set of tuples for fast lookup
    all_dimensions = {len(coord) for coord in squares}
    if len(all_dimensions) != 1:
        raise ValueError("All coordinates must have the same number of dimensions.")
    dimensions = all_dimensions.pop()
    if len(squares) < 2:
        return True
    coords = np.array(list(squares))

    # Define the 8-connected neighborhood offsets
    offsets = np.array(all_offsets(dimensions))

    # Check each coordinate
    for coord in coords:
        # Generate all 8-connected neighbors
        neighbors = coord + offsets

        # Check if any neighbor is in the original set
        if any(tuple(neighbor) in squares for neighbor in neighbors):
            continue
        else:
            return False

    return True


def is_connected(tree: KDTree, distance: float) -> bool:
    """
    Return true if the ``distance``-neighborhood graph is connected.

    Since we calculate the bins using this distance, this should be a good
    measure of connectedness.

    Args:
        tree: KDTree of the points.
        distance: The distance between two points that implies
           an edge in the graph (i.e., that they are neighbors).

    Returns:
        True if connected, False otherwise.
    """
    n_points = tree.data.shape[0]
    visited = set()
    to_visit = {0}  # Start with the first point

    while to_visit:
        current = to_visit.pop()
        visited.add(current)
        current_data = tree.data[current]
        neighbors = tree.query_radius([current_data], r=distance)[0]
        to_visit.update(set(neighbors) - visited)

    return len(visited) == n_points


def are_8_connected_kd(points: set[tuple]) -> bool:
    """
    Return true if the points are 8-connected, that is, every
    point is within 1 Chebyshev distance of another point.

    Since we calculate the bins using this distance, this should be a good
    measure of connectedness.

    Args:
        tree: KDTree of the points.
        distance: The distance between two points that implies
           an edge in the graph (i.e., that they are neighbors).

    Returns:
        True if connected, False otherwise.
    """
    # noinspection PyArgumentList
    tree = KDTree(list(points), metric="chebyshev")
    n_points = tree.data.shape[0]
    visited = set()
    to_visit = {0}  # Start with the first point

    while to_visit:
        current = to_visit.pop()
        visited.add(current)
        current_data = tree.data[current]
        neighbors = tree.query_radius([current_data], r=1)[0]
        to_visit.update(set(neighbors) - visited)

    return len(visited) == n_points


def covering_squares(points: np.ndarray, side_length: float) -> set[tuple]:
    # Calculate the indices of the squares for each point
    indices = np.floor(points / side_length).astype(int)

    # Use a set to count unique squares
    unique_squares: set[tuple] = set(map(tuple, indices))

    return unique_squares


def random_initial_conditions() -> np.ndarray:
    """
    Generate random initial conditions for the three-body problem.

    Returns:
        np.ndarray: Array of initial conditions [x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3].
    """
    positions = np.random.uniform(-1, 1, (3, 2))
    velocities = np.random.uniform(-0.1, 0.1, (3, 2))
    positions -= np.mean(positions, axis=0)
    velocities -= np.mean(velocities, axis=0)
    return np.ravel(np.column_stack((positions, velocities)))


def run_simulation(
    epsilon: float, t_span: tuple[float, float], num_points: int
) -> SimulationResult:
    """
    Run a single simulation of the modified three-body problem and calculate various metrics.

    Args:
        epsilon (float): Coupling strength for the y-dimension.
        t_span (tuple[float, float]): Time span for the simulation.
        num_points (int): Number of points to evaluate in the simulation.

    Returns:
        SimulationResult: Object containing simulation results and initial conditions.
    """
    status = -1
    while status != 0:
        y0 = random_initial_conditions()
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        sol = solve_ivp(
            modified_three_body,
            t_span,
            y0,
            args=(epsilon,),
            dense_output=True,
            t_eval=t_eval,
        )
        if sol.status != 0:
            logging.warning(
                f"Simulation failed with status {sol.status}. Retrying with new initial conditions."
            )
        status = sol.status

    # Estimate Lyapunov exponent
    lyapunov_exponent = estimate_lyapunov_exponent(
        modified_three_body, y0, t_span, num_points, epsilon, sol.y
    )

    # Calculate box-counting dimension
    points_for_dim = sol.y.T[
        num_points // 2 :, [0, 1, 4, 5, 8, 9]
    ]  # Only position coordinates, discarding transients
    dim, log_eps, log_num_boxes = box_counting_dim(points_for_dim)

    return SimulationResult(
        initial_conditions=y0.tolist(),
        dimension=dim,
        log_eps=log_eps,
        log_N=log_num_boxes,
        lyapunov=lyapunov_exponent,
    )


def main(args: SimulationParams) -> None:
    """
    Main function to run simulations and save results.

    Args:
        args (SimulationParams): Simulation parameters.
    """
    json_file = Path(args.output)

    try:
        with open(json_file, "r") as f:
            measured_dimensions = defaultdict(list, json.load(f))
    except FileNotFoundError:
        measured_dimensions = defaultdict(list)

    for epsilon in tqdm(args.epsilon, "Epsilon"):
        for _ in tqdm(range(args.trials), "Trials"):
            result = run_simulation(epsilon, (0, args.time), args.points)
            measured_dimensions[str(epsilon)].append(asdict(result))

    with open(json_file, "w") as f:
        json.dump(dict(measured_dimensions), f)


def get_args() -> SimulationParams:
    parser = argparse.ArgumentParser(
        description="Run Modified Three-Body Problem simulations."
        "Note: The output file will be overwritten if it already exists."
        "And I'm not confident in the mechanism for estimating the "
        "Lyapunov exponent, estimates something that increases with chaos, "
        "but I'm not sure it is the exponent."
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        nargs="+",
        default=[0, 0.1, 0.5, 1.0],
        help="List of epsilon values to simulate",
    )
    parser.add_argument(
        "--trials", type=int, default=5, help="Number of trials for each epsilon value"
    )
    parser.add_argument(
        "--time", type=float, default=1000, help="Total simulation time"
    )
    parser.add_argument(
        "--points",
        type=int,
        default=100000,
        help="Number of points to evaluate in the simulation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="measured_dimensions.json",
        help="Output JSON file name",
    )
    args = parser.parse_args()
    return SimulationParams(**vars(args))


if __name__ == "__main__":
    main(get_args())
