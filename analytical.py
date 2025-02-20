import numpy as np
import scipy.special as sp

def partition_function(N, h):
    """ Compute the partition function Z using binomial sum and closed form """
    # Binomial sum approach
    m_values = np.arange(-N, N+1, 2)  # Only even values of m occur
    binomial_coeffs = sp.comb(N, (N + m_values) // 2)  # Binomial coefficient
    Z_sum = np.sum(binomial_coeffs * np.exp(h * m_values))

    # Closed-form approach
    Z_closed = (2 * np.cosh(h)) ** N

    return Z_sum, Z_closed

def magnetization(N, h, Z):
    """ Compute the magnetization M """
    m_values = np.arange(-N, N+1, 2)
    binomial_coeffs = sp.comb(N, (N + m_values) // 2)
    M = np.sum(binomial_coeffs * m_values * np.exp(h * m_values)) / Z
    return M

def internal_energy(h, M):
    """ Compute the internal energy E """
    return -h * M

def heat_capacity(N, h, Z, M):
    """ Compute the heat capacity C_h """
    m_values = np.arange(-N, N+1, 2)
    binomial_coeffs = sp.comb(N, (N + m_values) // 2)
    avg_m2 = np.sum(binomial_coeffs * (m_values**2) * np.exp(h * m_values)) / Z
    C_h = h**2 * (avg_m2 - M**2)
    return C_h

# Define parameters
N = 16**2  # Number of spins
h = 2.  # External field

# Compute partition function
Z_sum, Z_closed = partition_function(N, h)

# Compute observables
M = magnetization(N, h, Z_sum) / N
E = internal_energy(h, M) / N
C_h = heat_capacity(N, h, Z_sum, M) / N

# Print results
print(f"Partition Function (Binomial Sum): {Z_sum}")
print(f"Partition Function (Closed Form): {Z_closed}")
print(f"Magnetization M: {M}")
print(f"Internal Energy E: {E}")
print(f"Heat Capacity C_h: {C_h}")
