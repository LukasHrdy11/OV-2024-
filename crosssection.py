import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import spherical_jn, spherical_yn


def potencial1(r, epsilon=5.9):
        return epsilon * ((1 / r)**12 - 2 * (1 / r)**6)

def numerov_phase_shift(E, potencial, l):
    my = (6.12 * (23.5)**2) / 2  
    hbarc = 23.5  
    krel = math.sqrt(2 * my * E) / hbarc  

    rmax = 5  
    h = 0.01  
    rgrid = np.linspace(0.01, rmax, int(rmax / h) + 1)  

    u0 = 0
    u1 = h**(l + 1)  
    
    for i in range(1, len(rgrid) - 1):
        k0 = 2 * my / hbarc**2 * (E - potencial(rgrid[i-1]) - hbarc**2 * l * (l + 1) / (2 * my * rgrid[i-1]**2))
        k1 = 2 * my / hbarc**2 * (E - potencial(rgrid[i]) - hbarc**2 * l * (l + 1) / (2 * my * rgrid[i]**2))
        k2 = 2 * my / hbarc**2 * (E - potencial(rgrid[i+1]) - hbarc**2 * l * (l + 1) / (2 * my * rgrid[i+1]**2))
        
        u2 = (2 * (1 - 5 / 12 * h**2 * k1) * u1 - (1 + 1 / 12 * h**2 * k0) * u0) / (1 + h**2 / 12 * k2)
        u0 = u1
        u1 = u2
    
    r0 = rgrid[-2]
    r1 = rgrid[-1]
    
    beta = u0 / u1 * r1 / r0
    
    bess_j_0 = spherical_jn(l, krel * r0)
    bess_n_0 = spherical_yn(l, krel * r0)
    
    bess_j_1 = spherical_jn(l, krel * r1)
    bess_n_1 = spherical_yn(l, krel * r1)
    
    tan_delta = (bess_j_0 - beta * bess_j_1) / (bess_n_0 - beta * bess_n_1)
    
    delta_l = math.atan(tan_delta) * 180 / math.pi
    if delta_l < 0:
        delta_l += 180
    return delta_l


def calculate_equation(E, potencial, l_max=6):
    my = (6.12 * (23.5)**2) / 2
    hbarc = 23.5
    k = math.sqrt(2 * my * E) / hbarc
    sum_value = 0

    for l in range(l_max + 1):
        #if l != 7:
            # continue
        delta_l = numerov_phase_shift(E, potencial, l)
        sum_value += (2 * l + 1) * (math.sin(math.radians(delta_l)))**2

    return (4 * math.pi / k**2) * sum_value

energy_values = np.linspace(0.1, 3.5, 50)  
equation_results = [calculate_equation(E, potencial1) for E in energy_values]

plt.figure(figsize=(8, 6))
plt.plot(energy_values, equation_results, color='c', marker='o')

plt.xlabel('E (meV)', fontsize = 18)
plt.ylabel(r'Celkový účinný průřez [$\rho^2$], $\rho = 3.57 \, \AA$' , fontsize=18)
plt.legend()
plt.tight_layout()
plt.show()
