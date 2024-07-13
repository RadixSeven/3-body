import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class SimulationResult:
    """Dataclass to store the results of a single simulation run."""

    initial_conditions: List[float]
    dimension: float
    log_eps: List[float]
    log_N: List[float]
    lyapunov: float


@dataclass
class SimulationParams:
    """Dataclass to store simulation parameters."""

    epsilon: List[float]
    trials: int
    time: float
    points: int
    output: str


def modified_three_body(
    _: float, state: np.ndarray, epsilon: float, gravity: float = 1
) -> np.ndarray:
    """
    Compute the derivatives for the modified three-body problem.

    Args:
        _: Current time (unused, but required by solve_ivp).
        state: Current state [x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3].
        epsilon: Coupling strength for the y-dimension.
        gravity: Gravitational constant. Defaults to 1.

    Returns:
        Derivatives [vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2, vx3, vy3, ax3, ay3].
    """
    # x1, y1 are the coordinates of the first body, and vx1, vy1 are the corresponding velocities
    x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3 = state

    # Distance is the same, the only change is the magnitude
    # of the force in the y-direction
    # rXYsq is the squared distance between bodies X and Y
    # rXY is the distance between bodies X and Y
    r12sq = (x1 - x2) ** 2 + (y1 - y2) ** 2
    r12 = np.sqrt(r12sq)
    r13sq = (x1 - x3) ** 2 + (y1 - y3) ** 2
    r13 = np.sqrt(r13sq)
    r23sq = (x2 - x3) ** 2 + (y2 - y3) ** 2
    r23 = np.sqrt(r23sq)

    # Magnitude of the force between bodies
    f12 = gravity / r12sq
    f13 = gravity / r13sq
    f23 = gravity / r23sq

    # Now, multiply by the appropriate trig function of the angle
    # to get components of the forces between the bodies
    # magnitude * sin(angle) = y-component
    # magnitude * cos(angle) = x-component
    # sin(angle) = opposite / hypotenuse = delta_y / r
    # cos(angle) = adjacent / hypotenuse = delta_x / r
    # I don't take the absolute value of the delta_x and delta_y
    # to preserve the direction of the force.
    f12_x = f12 * (x2 - x1) / r12
    f12_y = epsilon * f12 * (y2 - y1) / r12
    f13_x = f13 * (x3 - x1) / r13
    f13_y = epsilon * f13 * (y3 - y1) / r13
    f23_x = f23 * (x3 - x2) / r23
    f23_y = epsilon * f23 * (y3 - y2) / r23

    # Acceleration is force divided by mass (which is 1)
    # so sum the forces to get the acceleration
    ax1 = f12_x + f13_x
    ay1 = f12_y + f13_y

    ax2 = -f12_x + f23_x
    ay2 = -f12_y + f23_y

    ax3 = -f13_x - f23_x
    ay3 = -f13_y - f23_y

    return np.array([vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2, vx3, vy3, ax3, ay3])
