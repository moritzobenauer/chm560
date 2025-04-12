import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from box_muller import bma



dt = 0.001
T = 1000
times = np.arange(0, T, dt)

gamma = 0.5
kB = 1.0
T = 1.0

x = 1.0
p = 0.0

x_vals = []
p_vals = []

for t in times:
    x_vals.append(x)
    p_vals.append(p)

    p_half = p + (-x) * dt / 2
    x_new = x + p_half * dt

    p_new = p_half + (-x_new) * dt / 2

    kinetic_energy_t = 0.5 * p_new**2

    # np.random.normal for testing!
    # R = np.random.normal(0.0, np.sqrt(dt), size=1)

    # Using my own BMA instead of np.random.normal
    R = bma(1, 24) * np.sqrt(dt)

    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt((1 - c1**2) / (2 * gamma * dt))

    p_art = c1 * p_new + c2 * R

    kinetic_energy_tplus = 0.5 * p_art**2

    alpha = np.sqrt(kinetic_energy_tplus / kinetic_energy_t)
    p_new = alpha * p_new

    x, p = x_new, p_new[0]


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
    1.0 / np.sqrt(2 * np.pi) * np.exp(-1.0 / (2*kB * T) * momentum_linear**2),
    label=r"$\frac{1}{\sqrt{2\pi}} \exp(-0.5 p^2)$",
    color="k",
)
ax[1, 0].plot(
    momentum_linear,
    1.0 / np.sqrt(2 * np.pi) * np.exp(-1.0 / (2*kB * T) * momentum_linear**2),
    label=r"$\frac{1}{\sqrt{2\pi}} \exp(-0.5 x^2)$",
    color="k",
)
ax[1, 0].legend()

ax[1, 1].legend()
plt.tight_layout()
plt.savefig(f"vrescale_ho.png", dpi=150)
plt.show()

mean_kinetic = np.mean(np.array(p_vals) ** 2)
mean_potential = np.mean(np.array(x_vals) ** 2)
mean_energy = mean_kinetic + mean_potential

print(f"⟨K⟩ = {mean_kinetic:.3f}, ⟨U⟩ = {mean_potential:.3f}, ⟨H⟩ = {mean_energy:.3f}")
