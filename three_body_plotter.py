import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple
from scipy.integrate import solve_ivp

from three_body_common import SimulationResult, modified_three_body


@dataclass
class PlotterParams:
    """Dataclass to store plotter parameters."""

    input: str
    output: str
    plot_trajectories: bool
    output_format: str


def load_data(json_file: str) -> Dict[str, List[SimulationResult]]:
    """
    Load simulation results from a JSON file.

    Args:
        json_file (str): Path to the input JSON file.

    Returns:
        Dict[str, List[SimulationResult]]: Dictionary of simulation results, keyed by epsilon values.

    Raises:
        FileNotFoundError: If the input file is not found.
    """
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
        return {
            epsilon: [SimulationResult(**result) for result in results]
            for epsilon, results in data.items()
        }
    except FileNotFoundError:
        print(f"Error: Input file '{json_file}' not found.")
        raise


def plot_trajectory(
    initial_conditions: List[float],
    epsilon: float,
    t_span: Tuple[float, float],
    num_points: int,
    output_prefix: str,
    output_format: str,
) -> None:
    """
    Plot the trajectory of the three bodies for a given set of initial conditions.

    Args:
        initial_conditions: Initial conditions for the simulation.
        epsilon: Coupling strength for the y-dimension.
        t_span: Time span for the simulation.
        num_points: Number of points to evaluate in the simulation.
        output_prefix: Prefix for the output file name.
        output_format: The output file format ("svg" or "png").
    """
    sol = solve_ivp(
        modified_three_body,
        t_span,
        initial_conditions,
        args=(epsilon,),
        dense_output=True,
        t_eval=np.linspace(t_span[0], t_span[1], num_points),
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(sol.y[0], sol.y[1], label="Body 1")
    ax.plot(sol.y[4], sol.y[5], label="Body 2")
    ax.plot(sol.y[8], sol.y[9], label="Body 3")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Three-Body Trajectories (ε = {epsilon})")
    ax.legend()
    ax.grid(True)
    plt.savefig(
        f"{output_prefix}_trajectory_epsilon_{epsilon}.{output_format}",
        format=output_format,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_dimensions_and_lyapunov(
    data: Dict[str, List[SimulationResult]], output_prefix: str, output_format: str
) -> None:
    """
    Plot boxplots of attractor dimensions and Lyapunov exponents.

    Args:
        data: Dictionary of simulation results.
        output_prefix: Prefix for the output file name.
        output_format: The output file format ("svg" or "png").
    """
    epsilon_values = sorted(float(eps) for eps in data.keys())
    dimensions = [[r.dimension for r in data[str(eps)]] for eps in epsilon_values]
    lyapunovs = [[r.lyapunov for r in data[str(eps)]] for eps in epsilon_values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.boxplot(dimensions, tick_labels=epsilon_values)
    ax1.set_xlabel("Epsilon")
    ax1.set_ylabel("Estimated Attractor Dimension")
    ax1.set_title("Attractor Dimension vs. Epsilon")
    ax1.grid(True)

    ax2.boxplot(lyapunovs, tick_labels=epsilon_values)
    ax2.set_xlabel("Epsilon")
    ax2.set_ylabel("Lyapunov Exponent")
    ax2.set_title("Lyapunov Exponent vs. Epsilon")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(
        f"{output_prefix}_dimensions_lyapunov.{output_format}",
        format=output_format,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_dimension_lyapunov_vs_epsilon(
    data: Dict[str, List[SimulationResult]], output_prefix: str, output_format: str
) -> None:
    """
    Plot mean dimension and Lyapunov exponent vs epsilon with error bars.

    Args:
        data: Dictionary of simulation results.
        output_prefix: Prefix for the output file name.
        output_format: The output file format ("svg" or "png").
    """
    epsilon_values = sorted(float(eps) for eps in data.keys())
    dimensions = [[r.dimension for r in data[str(eps)]] for eps in epsilon_values]
    lyapunovs = [[r.lyapunov for r in data[str(eps)]] for eps in epsilon_values]

    mean_dimensions = [np.mean(d) for d in dimensions]
    std_dimensions = [np.std(d) for d in dimensions]
    mean_lyapunovs = [np.mean(l) for l in lyapunovs]
    std_lyapunovs = [np.std(l) for l in lyapunovs]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.errorbar(
        epsilon_values,
        mean_dimensions,
        yerr=std_dimensions,
        fmt="o-",
        label="Dimension",
    )
    ax1.set_xlabel("Epsilon")
    ax1.set_ylabel("Estimated Attractor Dimension")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.errorbar(
        epsilon_values,
        mean_lyapunovs,
        yerr=std_lyapunovs,
        fmt="s-",
        color="r",
        label="Lyapunov Exponent",
    )
    ax2.set_ylabel("Lyapunov Exponent")
    ax2.tick_params(axis="y")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Dimension and Lyapunov Exponent vs. Epsilon")
    plt.grid(True)
    plt.savefig(
        f"{output_prefix}_dimension_lyapunov_vs_epsilon.{output_format}",
        format=output_format,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_dimensions_and_metrics(params: PlotterParams) -> None:
    """
    Main function to plot all metrics from the simulation results.

    Args:
        params (PlotterParams): Plotter parameters including input and output file names.
    """
    data = load_data(params.input)
    plot_dimensions_and_lyapunov(data, params.output, params.output_format)
    plot_dimension_lyapunov_vs_epsilon(data, params.output, params.output_format)

    if params.plot_trajectories:
        # Plot trajectory for the simulation with the highest Lyapunov exponent for each epsilon
        for epsilon, results in data.items():
            max_lyap_result = max(results, key=lambda r: r.lyapunov)
            plot_trajectory(
                max_lyap_result.initial_conditions,
                max_lyap_result.epsilon,
                (max_lyap_result.time_start, max_lyap_result.time_stop),
                max_lyap_result.num_points,
                params.output + "_max_exponent",
                params.output_format,
            )
            max_dim_result = max(results, key=lambda r: r.dimension)
            plot_trajectory(
                max_dim_result.initial_conditions,
                max_dim_result.epsilon,
                (max_dim_result.time_start, max_dim_result.time_stop),
                max_dim_result.num_points,
                params.output + "_max_dimension",
                params.output_format,
            )


def main(args: argparse.Namespace) -> None:
    """
    Main function to run the plotter.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    params = PlotterParams(
        input=args.input,
        output=args.output,
        plot_trajectories=args.plot_trajectories,
        output_format=args.output_format,
    )
    plot_dimensions_and_metrics(params)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Three-Body Problem simulation results"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="measured_dimensions.json",
        help="Input JSON file name",
    )
    parser.add_argument(
        "--output", type=str, default="dimension_plot", help="Output SVG file prefix"
    )
    parser.add_argument(
        "--plot-trajectories",
        action="store_true",
        help="Plot trajectories for interesting cases",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["svg", "png"],
        default="png",
        help="Output file format",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
