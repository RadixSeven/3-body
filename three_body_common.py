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
    _: float, state: np.ndarray, gravity_y_strength: float, gravity: float = 1
) -> np.ndarray:
    """
    Compute the derivatives for the modified three-body problem.

    Args:
        _: Current time (unused, but required by solve_ivp).
        state: Current state [x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3].
        gravity_y_strength: Coupling strength for the y-dimension.
        gravity: Gravitational constant. Defaults to 1.

    Returns:
        Derivatives [vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2, vx3, vy3, ax3, ay3].
    """
    # x1, y1 are the coordinates of the first body, and vx1, vy1 are the corresponding velocities
    x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3 = state

    # Distance is the same, the only change is the magnitude
    # of the force in the y-direction
    # rXY is the distance between bodies X and Y
    r12 = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    r13 = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
    r23 = np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)

    # Calculate the cubes (r**3) of the distances because
    # of the cubic term that appears later on
    r12_cube = r12**3
    r13_cube = r13**3
    r23_cube = r23**3

    # Magnitude of the force between bodies
    # f12 = gravity / (r12 ** 2)
    # f13 = gravity / (r13 ** 2)
    # f23 = gravity / (r23 ** 2)
    #
    # Now, multiply by the appropriate trig function of the angle
    # to get components of the forces between the bodies
    # magnitude * sin(angle) = y-component
    # magnitude * cos(angle) = x-component
    # sin(angle) = opposite / hypotenuse = delta_y / r
    # cos(angle) = adjacent / hypotenuse = delta_x / r
    # I don't take the absolute value of the delta_x and delta_y
    # to preserve the direction of the force.
    #
    # When you calculate the force components, you get an
    # r**3 in the denominator, to change the distance term
    # in the numerator into the fraction of the force along
    # the appropriate axis.
    #
    # f12_x = f12 * (x2 - x1) / r12 and substitute f12
    #       = (gravity / (r12 ** 2)) * (x2 - x1) / r12
    #       = gravity * (x2 - x1) / (r12 ** 3)
    f12_x = gravity * (x2 - x1) / r12_cube
    # Similarly for y, but multiply by gravity_y_strength
    f12_y = gravity_y_strength * gravity * (y2 - y1) / r12_cube
    f13_x = gravity * (x3 - x1) / r13_cube
    f13_y = gravity_y_strength * gravity * (y3 - y1) / r13_cube
    f23_x = gravity * (x3 - x2) / r23_cube
    f23_y = gravity_y_strength * gravity * (y3 - y2) / r23_cube

    # Acceleration is force divided by mass (which is 1)
    # so sum the forces to get the acceleration
    ax1 = f12_x + f13_x
    ay1 = f12_y + f13_y

    ax2 = -f12_x + f23_x
    ay2 = -f12_y + f23_y

    ax3 = -f13_x - f23_x
    ay3 = -f13_y - f23_y

    return np.array([vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2, vx3, vy3, ax3, ay3])
