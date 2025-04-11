import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numba import njit


@njit
def update_thermostats(
    momenta: np.array, kB: float, T: float, Q: float, dt: float, p: float
) -> np.array:

    length = len(momenta)
    if length == 1:
        momenta[0] += (p**2 - kB * T) * dt

    else:

        for i in range(length):
            if i == length - 1:
                momenta[i] += (momenta[i - 1] ** 2 / Q - kB * T) * dt
            elif i == 0:
                momenta[i] += (p**2 - kB * T - momenta[i + 1] / Q * momenta[i]) * dt
            elif i == length:
                continue
            else:
                momenta[i] += (
                    momenta[i - 1] ** 2 / Q - kB * T - momenta[i + 1] / Q * momenta[i]
                ) * dt

    etas = 1.0 / Q * (momenta)
    return etas


dt = 0.001
TIME = 10000
times = np.arange(0, TIME, dt)

kB = 1.0
T = 1.0
Q = 0.5

x = 1.0
p = 0.0
NUM_THERMOSTATS = 0
TURNON = 0
MOMENTA_THERMOSTAT = np.zeros(NUM_THERMOSTATS, dtype=float)
ETAS = 1.0 / Q * (MOMENTA_THERMOSTAT) * TURNON


x_vals = np.array([], dtype=float)
p_vals = np.array([], dtype=float)


@njit
def run_sim(
    update_thermostats,
    dt,
    TIME,
    times,
    kB,
    Q,
    x,
    p,
    TURNON,
    MOMENTA_THERMOSTAT,
    ETAS,
    x_vals,
    p_vals,
):

    steps = int(TIME / dt)
    x_vals = np.zeros(steps)
    p_vals = np.zeros(steps)

    for i in range(steps):

        x_vals[i] = x
        p_vals[i] = p

        p_half = p + (-x - p * ETAS[0]) * dt / 2
        x_new = x + p_half * dt

        ETAS = update_thermostats(MOMENTA_THERMOSTAT, kB, T, Q, dt, p) * TURNON

        p_new = p_half + (-x_new - p * ETAS[0]) * dt / 2

        x, p = x_new, p_new

        if i % 10000 == 0:
            print("Time", i * dt)

    return x_vals, p_vals


(x_vals, p_vals) = run_sim(
    update_thermostats,
    dt,
    TIME,
    times,
    kB,
    Q,
    x,
    p,
    TURNON,
    MOMENTA_THERMOSTAT,
    ETAS,
    x_vals,
    p_vals,
)

print(x_vals)

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

circle = plt.Circle((0, 0), 1.0, color="snow", fill=False, lw=3, linestyle="dashed")
ax[0, 0].scatter(x_vals, p_vals, alpha=0.01, edgecolors="white")
ax[0, 0].add_patch(circle)
ax[0, 0].set_xlabel(r"Position $x$")
ax[0, 0].set_ylabel(r"Momentum $p$")
ax[0, 0].set_title(r"Phase Space $\Gamma$")
ax[0, 0].set_xlim(-3, 3)
ax[0, 0].set_ylim(-3, 3)

ax[0, 1].plot(times[::1000], x_vals[::1000], label=r"$x$")
ax[0, 1].plot(times[::1000], p_vals[::1000], label=r"$p$")
ax[0, 1].set_title(r"Trajectory versus time")
ax[0, 1].legend()
ax[0, 1].set_xlabel(r"Time $t$")


ax[1, 0].hist(x_vals, bins=50, density=True, alpha=0.7)
ax[1, 0].set_title(r"Position Probability Distribution")
ax[1, 1].set_title(r"Momentum Probability Distribution")

ax[1, 0].set_xlabel(r"Position $x$")
ax[1, 0].set_ylabel(r"$f(x)$")

ax[1, 1].set_xlabel(r"Momentum $p$")
ax[1, 1].set_ylabel(r"$f(p)$")


ax[1, 1].hist(p_vals, bins=50, density=True, alpha=0.7)
momentum_linear = np.linspace(-5, 5, 100)
ax[1, 1].plot(
    momentum_linear,
    1.0 / np.sqrt(2 * np.pi) * np.exp(-1.0 / (kB * T) * momentum_linear**2),
    label="Boltzmann distribution",
    color="k",
)
ax[1, 0].plot(
    momentum_linear,
    1.0 / np.sqrt(2 * np.pi) * np.exp(-1.0 / (kB * T) * momentum_linear**2),
    label="Boltzmann distribution",
    color="k",
)
ax[1, 0].legend()

ax[1, 1].legend()
plt.tight_layout()
plt.savefig(f"thermostat_{TURNON}_{NUM_THERMOSTATS}_ho.png", dpi=150)
plt.show()

mean_kinetic = np.mean(np.array(p_vals) ** 2)
mean_potential = np.mean(np.array(x_vals) ** 2)
mean_energy = mean_kinetic + mean_potential

print(f"⟨K⟩ = {mean_kinetic:.3f}, ⟨U⟩ = {mean_potential:.3f}, ⟨H⟩ = {mean_energy:.3f}")
