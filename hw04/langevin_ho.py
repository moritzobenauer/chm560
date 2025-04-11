import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from box_muller import bma


dt = 0.001
T = 3000
times = np.arange(0, T, dt)

gamma=0.5
kB = 1.0            
T = 1.0               
sqrt_coeff = np.sqrt(gamma * kB * T * dt)

x = 1.0 
p = 0.0  

x_vals = []
p_vals = []

for t in times:
    x_vals.append(x)
    p_vals.append(p)
    
    # R1 = np.random.normal(0, 1)
    # R2 = np.random.normal(0, 1)

    R1 = bma(1, 24)
    R2 = bma(1, 24)

    c1 = np.exp(-gamma * dt/2.)
    c2 = np.sqrt((1 - c1**2)/(gamma*dt))    

    p_plus = c1*p + c2 * R1 * sqrt_coeff


    x_new = x + p_plus * dt + dt**2*(-p)/2.

    p_minus = p_plus + dt/2. * (-x - x_new)

    p_new = c1*p_minus + c2 * R2 * sqrt_coeff
    
    x, p = x_new, p_new

fig, ax = plt.subplots(2,2,figsize=(10,10))

circle = plt.Circle((0, 0), 1.0, color='snow', fill=False, lw=3, linestyle='dashed')
ax[0,0].scatter(x_vals, p_vals, alpha=0.01, edgecolors='white')
ax[0,0].add_patch(circle)
ax[0,0].set_xlabel(r'Position $x$')
ax[0,0].set_ylabel(r'Momentum $p$')
ax[0,0].set_title(r'Phase Space $\Gamma$')
ax[0,0].set_xlim(-3,3)
ax[0,0].set_ylim(-3,3)

ax[0,1].plot(times,x_vals, label=r'$x$')
ax[0,1].plot(times,p_vals, label=r'$p$')
ax[0,1].set_title(r'Trajectory versus time')
ax[0,1].legend()
ax[0,1].set_xlabel(r'Time $t$')


ax[1,0].hist(x_vals, bins=50, density=True, alpha=0.7)
ax[1,0].set_title(r'Position Probability Distribution')
ax[1,1].set_title(r'Momentum Probability Distribution')

ax[1,0].set_xlabel(r'Position $x$')
ax[1,0].set_ylabel(r'$f(x)$')

ax[1,1].set_xlabel(r'Momentum $p$')
ax[1,1].set_ylabel(r'$f(p)$')


ax[1,1].hist(p_vals, bins=50, density=True, alpha=0.7)
momentum_linear = np.linspace(-5,5,100)
ax[1,1].plot(momentum_linear, 1./np.sqrt(2*np.pi)*np.exp(-1./(kB*T) * momentum_linear**2), label='Boltzmann distribution', color='k')
ax[1,0].plot(momentum_linear, 1./np.sqrt(2*np.pi)*np.exp(-1./(kB*T) * momentum_linear**2), label='Boltzmann distribution', color='k')
ax[1,0].legend()

ax[1,1].legend()
plt.tight_layout()
plt.savefig('langevin_ho.png', dpi=150)
# plt.show()

mean_kinetic = np.mean(np.array(p_vals)**2)
mean_potential = np.mean(np.array(x_vals)**2)
mean_energy = mean_kinetic + mean_potential

print(f"⟨K⟩ = {mean_kinetic:.3f}, ⟨U⟩ = {mean_potential:.3f}, ⟨H⟩ = {mean_energy:.3f}")

