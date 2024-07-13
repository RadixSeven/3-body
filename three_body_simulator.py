import numpy as np
from scipy.integrate import solve_ivp
from sklearn.neighbors import KDTree
import json
from pathlib import Path
import argparse
from collections import defaultdict
from dataclasses import asdict

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


def box_counting_dim(
    points: np.ndarray, n_samples: int = 20
) -> tuple[float, list[float], list[float]]:
    """
    Estimate the box-counting dimension of a set of points.

    Args:
        points: Array of points around which boxes will be placed.
        n_samples: Number of epsilon values to sample. Defaults to 20.

    Returns:
        Estimated dimension, log(1/epsilon) values, log(num_points) values.
    """
    tree = KDTree(points)
    eps_min, eps_max = estimate_epsilon_range(points)
    epsilons = np.logspace(np.log10(eps_min), np.log10(eps_max), n_samples)

    log_eps = []
    log_n = []

    for epsilon in epsilons:
        if not check_connectedness(tree, epsilon):
            break

        num_points = len(tree.query_radius([[0] * points.shape[1]], r=epsilon)[0])
        log_eps.append(np.log(1 / epsilon))
        log_n.append(np.log(num_points))

    # Perform linear fit on the connected portion
    slope, _ = np.polyfit(log_eps, log_n, 1)

    return slope, log_eps, log_n


def estimate_epsilon_range(points: np.ndarray) -> tuple[float, float]:
    """
    Estimate a suitable range for epsilon values based on the data.

    Args:
        points: Array of points.

    Returns:
        tuple[float, float]: Minimum and maximum epsilon values.
    """
    distances = KDTree(points).query(points, k=2)[0][
        :, 1
    ]  # Distances to nearest neighbors
    min_distance = np.min(distances)
    max_distance = np.max(np.ptp(points, axis=0))  # Maximum extent of data
    return min_distance / 2, max_distance


def check_connectedness(tree: KDTree, epsilon: float) -> bool:
    """
    Check if the epsilon-neighborhood graph is connected.

    Args:
        tree (KDTree): KDTree of the points.
        epsilon (float): Current epsilon value.

    Returns:
        bool: True if connected, False otherwise.
    """
    n_points = tree.data.shape[0]
    visited = set()
    to_visit = {0}  # Start with the first point

    while to_visit:
        current = to_visit.pop()
        visited.add(current)
        neighbors = tree.query_radius([tree.data[current]], r=epsilon)[0]
        to_visit.update(set(neighbors) - visited)

    return len(visited) == n_points


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

    for epsilon in args.epsilon:
        for _ in range(args.trials):
            result = run_simulation(epsilon, (0, args.time), args.points)
            measured_dimensions[str(epsilon)].append(asdict(result))

        avg_dim = np.mean([r["dimension"] for r in measured_dimensions[str(epsilon)]])
        avg_exponent = np.mean(
            [r["lyapunov"] for r in measured_dimensions[str(epsilon)]]
        )
        print(
            f"Epsilon = {epsilon}: Avg. Dimension = {avg_dim:.3f}, Avg. Lyapunov Exponent = {avg_exponent:.3f}"
        )

    with open(json_file, "w") as f:
        json.dump(dict(measured_dimensions), f)


def get_args():
    parser = argparse.ArgumentParser(description="Run Three-Body Problem simulations")
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
    return parser.parse_args()


if __name__ == "__main__":
    main(SimulationParams(**vars(get_args())))
