import math
import numpy as np
from scipy.optimize import minimize

energy_values = [1, 5, 10, 25, 50, 100, 150, 200]
energy_values = [x / 2 for x in energy_values]
observed_phase_shifts = [147.747, 118.178, 102.611, 80.63, 62.77, 43.23, 30.72, 21.22]

def calculate_phase_shift(E, potencial, l):
    def numerov(E, potencial, l):
        u0 = 0
        u1 = h**(l+1)
        x = int(rmax/h)
        u2 = 0
        for i in range (0,x-2):
            k0 = 2*my/hbarc**2*(E - potencial(0+i*h)-hbarc**2*l*(l+1)/(2*my*rmax**2))
            k1 = 2*my/hbarc**2*(E - potencial(h+i*h)-hbarc**2*l*(l+1)/(2*my*rmax**2))
            k2 = 2*my/hbarc**2*(E - potencial(2*h+i*h)-hbarc**2*l*(l+1)/(2*my*rmax**2))
            u2 = (2*(1-5/12*pow(h,2)*k1)*u1-(1+1/12*pow(h,2)*k0)*u0)/(1+pow(h,2)/12*k2)
            u0 = u1
            u1 = u2
                
        return u0, u1

    my = 938/2
    hbarc = 197.32697
    k = math.sqrt(2*my*E)/hbarc
    rmax = 30
    h = 0.01
    r2 = rmax - h
    r1 = rmax - 2*h
    j1 = math.sin(k * r1) / (k*r1)
    n1 = -math.cos(k * r1) / (k * r1)
    j2 = math.sin(k * r2) / (k*r2)
    n2 = -math.cos(k * r2) / (k * r2)
    u1, u2 = numerov(E, potencial, l)
    b = u1/u2 * r2/r1
    up = j1- b*j2
    down = n1- b*n2

    delta_l = math.atan(up/down) * 180/(math.pi)
    if delta_l < 0:
        delta_l = delta_l + 180
    return delta_l

def fitness_function(params):
    def potential(r):
        V = params[0] * np.exp(-params[1] * r**2)  + params[2] * np.exp(-params[3] * r**2) + params[4] * np.exp(-params[5] * r**2)+params[6] * np.exp(-params[7] * r**2)
        return V
    
    calculated_phase_shifts = np.array([calculate_phase_shift(energy, potential, 0) for energy in energy_values])
    return np.sum((calculated_phase_shifts - observed_phase_shifts) ** 2)

initial_guess = np.array([200, 1.487, 160, 0.6, 500, 1.2, 1500, 3.2])  

result = minimize(fitness_function, initial_guess, method='Nelder-Mead')

best_params = result.x
best_fitness = result.fun

print("Nejlepší nalezené parametry:", best_params)
print("Fitness nejlepšího řešení:", best_fitness)

def best_potential(r):
    return best_params[0] * np.exp(-best_params[1] * r**2)  + best_params[2] * np.exp(-best_params[3] * r**2) + best_params[4] * np.exp(-best_params[5] * r**2) + best_params[6] * np.exp(-best_params[7] * r**2) 

calculated_phase_shifts = [calculate_phase_shift(energy, best_potential, 0) for energy in energy_values]

for idx, phase_shift in enumerate(calculated_phase_shifts):
    energy = energy_values[idx]
    print(f"Energy: {energy}, Calculated Phase shift: {phase_shift}")
