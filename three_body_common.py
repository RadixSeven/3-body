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
    energy: List[float]
    angular_momentum: List[float]


@dataclass
class SimulationParams:
    """Dataclass to store simulation parameters."""

    epsilon: List[float]
    trials: int
    time: float
    points: int
    output: str


def modified_three_body(
    _: float, state: np.ndarray, epsilon: float, G: float = 1
) -> np.ndarray:
    """
    Compute the derivatives for the modified three-body problem.

    Args:
        _ (float): Current time (unused, but required by solve_ivp).
        state (np.ndarray): Current state [x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3].
        epsilon (float): Coupling strength for the y-dimension.
        G (float): Gravitational constant. Defaults to 1.

    Returns:
        np.ndarray: Derivatives [vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2, vx3, vy3, ax3, ay3].
    """
    x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3 = state

    r12 = np.sqrt((x1 - x2) ** 2 + (epsilon * (y1 - y2)) ** 2)
    r13 = np.sqrt((x1 - x3) ** 2 + (epsilon * (y1 - y3)) ** 2)
    r23 = np.sqrt((x2 - x3) ** 2 + (epsilon * (y2 - y3)) ** 2)

    ax1 = G * (x2 - x1) / r12**3 + G * (x3 - x1) / r13**3
    ay1 = epsilon * (G * (y2 - y1) / r12**3 + G * (y3 - y1) / r13**3)

    ax2 = G * (x1 - x2) / r12**3 + G * (x3 - x2) / r23**3
    ay2 = epsilon * (G * (y1 - y2) / r12**3 + G * (y3 - y2) / r23**3)

    ax3 = G * (x1 - x3) / r13**3 + G * (x2 - x3) / r23**3
    ay3 = epsilon * (G * (y1 - y3) / r13**3 + G * (y2 - y3) / r23**3)

    return np.array([vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2, vx3, vy3, ax3, ay3])


def calculate_energy(state: np.ndarray, epsilon: float, G: float = 1) -> float:
    """
    Calculate the total energy of the system.

    Args:
        state (np.ndarray): Current state [x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3].
        epsilon (float): Coupling strength for the y-dimension.
        G (float): Gravitational constant. Defaults to 1.

    Returns:
        float: Total energy of the system.
    """
    x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3 = state

    # Kinetic energy
    KE = 0.5 * (vx1**2 + vy1**2 + vx2**2 + vy2**2 + vx3**2 + vy3**2)

    # Potential energy
    r12 = np.sqrt((x1 - x2) ** 2 + (epsilon * (y1 - y2)) ** 2)
    r13 = np.sqrt((x1 - x3) ** 2 + (epsilon * (y1 - y3)) ** 2)
    r23 = np.sqrt((x2 - x3) ** 2 + (epsilon * (y2 - y3)) ** 2)
    PE = -G * (1 / r12 + 1 / r13 + 1 / r23)

    return KE + PE


def calculate_angular_momentum(state: np.ndarray) -> float:
    """
    Calculate the total angular momentum of the system.

    Args:
        state (np.ndarray): Current state [x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3].

    Returns:
        float: Total angular momentum of the system.
    """
    x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3 = state
    L1 = x1 * vy1 - y1 * vx1
    L2 = x2 * vy2 - y2 * vx2
    L3 = x3 * vy3 - y3 * vx3
    return L1 + L2 + L3
