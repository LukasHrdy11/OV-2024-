import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

energy_values = [1, 5, 10, 25, 50, 100, 150, 200]
energy_values = [x / 2 for x in energy_values]
observed_phase_shifts_degrees = [147.747, 118.178, 102.611, 80.63, 62.77, 43.23, 30.72, 21.22]
observed_phase_shifts_radians = [math.radians(number) for number in observed_phase_shifts_degrees]
observed_phase_shifts = [math.tan(number) for number in observed_phase_shifts_radians]

def calculate_phase_shift(E, potencial, l):
    def numerov(E, potencial, l):
        my = 938 / 2
        hbarc = 197.32697
        rmax = 30
        h = 0.1
        u0 = 0
        u1 = h ** (l + 1)
        x = int(rmax / h)
        for i in range(0, x - 2):
            r = (i + 1) * h
            k0 = 2 * my / hbarc**2 * (E - potencial(r) - hbarc**2 * l * (l + 1) / (2 * my * r**2))
            k1 = 2 * my / hbarc**2 * (E - potencial(r + h) - hbarc**2 * l * (l + 1) / (2 * my * r**2))
            k2 = 2 * my / hbarc**2 * (E - potencial(r + 2 * h) - hbarc**2 * l * (l + 1) / (2 * my * r**2))
            u2 = (2 * (1 - 5 / 12 * h**2 * k1) * u1 - (1 + 1 / 12 * h**2 * k0) * u0) / (1 + h**2 / 12 * k2)
            u0 = u1
            u1 = u2
        return u0, u1

    my = 938/2
    hbarc = 197.32697
    k = math.sqrt(2*my*E)/hbarc
    rmax = 30
    h = 0.1
    r2 = rmax - h
    r1 = rmax - 2*h
    j1 = (math.sin(k * r1) / (k * r1)**2 - math.cos(k * r1) / (k * r1))
    n1 = (-math.cos(k * r1) / (k * r1)**2 - math.sin(k * r1) / (k * r1))
    j2 = (math.sin(k * r2) / (k * r2)**2 - math.cos(k * r2) / (k * r2))
    n2 = (-math.cos(k * r2) / (k * r2)**2 - math.sin(k * r2) / (k * r2))
    u1, u2 = numerov(E, potencial, l)
    b = u1/u2 * r2/r1
    up = j1 - b * j2
    down = n1 - b * n2

    delta_l = up / down

    return delta_l

def fitness_function(params):
    def potential(r):
        return (params[0] * np.exp(-params[1] * r**2) +
                params[2] * np.exp(-params[3] * r**2) +
                params[4] * np.exp(-params[5] * r**2) +
                params[6] * np.exp(-params[7] * r**2) + 
                params[8] * np.exp(-params[9] * r**2))

    calculated_phase_shifts = np.array([calculate_phase_shift(energy, potential, 1) for energy in energy_values])
    return np.sum((np.abs(calculated_phase_shifts) - np.abs(observed_phase_shifts)) ** 2)


initial_guess = np.array([50, 0.5, -30, 1.0, 20, 2.0, -10, 3.0, 5, 4.0])  
best_params = initial_guess
iteration = 0
max_iterations = 3

for iteration in range(max_iterations):
    result = minimize(fitness_function, best_params, method='Nelder-Mead')
    best_params = result.x
    best_fitness = result.fun

    print(f"Iteration {iteration + 1}:")
    print("Best parameters:", ", ".join([f"{param:.8e}" for param in best_params]))
    print("Best fitness:", best_fitness)


def best_potential(r):
    return (best_params[0] * np.exp(-best_params[1] * r**2) +
            best_params[2] * np.exp(-best_params[3] * r**2) +
            best_params[4] * np.exp(-best_params[5] * r**2) +
            best_params[6] * np.exp(-best_params[7] * r**2) + 
            best_params[8] * np.exp(-best_params[9] * r**2))

calculated_phase_shifts = [math.degrees(math.atan(calculate_phase_shift(energy, best_potential, 1))) for energy in energy_values]


for i in range(len(calculated_phase_shifts)):
    if calculated_phase_shifts[i] < 0:
        calculated_phase_shifts[i] += 180


calculated_phase_shifts[-1] -= 180


energy_values = [0] + energy_values
calculated_phase_shifts = [180] + calculated_phase_shifts


observed_phase_shifts_degrees_with_placeholder = [None] + observed_phase_shifts_degrees



plt.figure(figsize=(8, 6))
plt.plot(energy_values, calculated_phase_shifts, label='Fit', color='blue', marker='o', linestyle='-', linewidth=2)
plt.scatter(energy_values[1:], observed_phase_shifts_degrees, label='Experiment', color='red', marker='x', s=50)


plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)


plt.xlabel("E (MeV)",  fontsize=18)
plt.ylabel("Fázový posun (°)",  fontsize=18)


plt.xlim(-1.5, 110) 
plt.ylim(-15, 190)  

plt.yticks(range(-15, 181, 15))
plt.xticks(range(0, 110, 10))

plt.grid(False)

plt.legend(loc='upper right', frameon=False)
plt.legend(fontsize=16)

plt.show()

