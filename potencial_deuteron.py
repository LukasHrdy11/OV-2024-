import math
import numpy as np
from scipy.optimize import minimize
import os
import matplotlib.pyplot as plt

# Načtení fázových posunů ze souborů
def load_phase_shifts():
    phase_shifts = {}
    base_path = os.path.dirname(__file__)
    file_names = ['S0I1', 'S1I0', 'S1I1', 'S0I0']
    for file_name in file_names:
        with open(os.path.join(base_path, file_name), 'r') as f:
            lines = f.readlines()
            energies = []
            shifts = []
            for line in lines:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                energies.append(float(parts[0]))
                shifts.append(float(parts[1]))
            phase_shifts[file_name] = (energies, shifts)
    return phase_shifts

# Definice výpočtu fázového posunu
def calculate_phase_shift(E, potencial, l):
    def numerov(E, potencial, l):
        u0 = 0
        u1 = h**(l+1)
        x = int(rmax/h)
        u2 = 0
        for i in range(0, x-2):
            k0 = 2*my/hbarc**2*(E - potencial(0+i*h) - hbarc**2*l*(l+1)/(2*my*rmax**2))
            k1 = 2*my/hbarc**2*(E - potencial(h+i*h) - hbarc**2*l*(l+1)/(2*my*rmax**2))
            k2 = 2*my/hbarc**2*(E - potencial(2*h+i*h) - hbarc**2*l*(l+1)/(2*my*rmax**2))
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
    up = j1 - b * j2
    down = n1 - b * n2
    delta_l = math.atan(up/down) * 180/(math.pi)
    if delta_l < 0:
        delta_l += 180
    return delta_l

# funkce fitness
def fitness_function(params, energy_values, observed_phase_shifts):
    def potential(r):
        V = params[0] * np.exp(-params[1] * r**2)
        return V
    calculated_phase_shifts = np.array([calculate_phase_shift(energy, potential, 0) for energy in energy_values])
    return np.sum((calculated_phase_shifts - observed_phase_shifts) ** 2)

# Spočítání potenciálu pro daný kanál 
def calculate_potential_for_channel(energy_values, observed_phase_shifts):
    initial_guess = np.array([-100, 1.0])

    def callback(params):
        current_fitness = fitness_function(params, energy_values, observed_phase_shifts)
        print(f"Aktuální parametry: {params}, Fitness: {current_fitness}")

    result = minimize(fitness_function, initial_guess, args=(energy_values, observed_phase_shifts), 
                      method='Nelder-Mead', callback=callback, options={'maxiter': 50})
    best_params = result.x
    best_fitness = result.fun

    print("Nejlepší nalezené parametry:", best_params)
    print("Fitness nejlepšího řešení:", best_fitness)

    def best_potential(r):
        return best_params[0] * np.exp(-best_params[1] * r**2)
    
    return best_params, best_potential

# Složení celkového potenciálu
def combine_potentials(params_list):
    def combined_potential(r):
        total_potential = sum(p[0] * np.exp(-p[1] * r**2) for p in params_list)
        return total_potential / len(params_list)
    return combined_potential

# Hlavní část skriptu
phase_shifts_data = load_phase_shifts()
all_params = []

for file_name, (energy_values, observed_phase_shifts) in phase_shifts_data.items():
    print(f"Zpracovávám soubor: {file_name}")
    best_params, _ = calculate_potential_for_channel(energy_values, observed_phase_shifts)
    all_params.append(best_params)

combined_potential = combine_potentials(all_params)

# Vykreslení kombinovaného potenciálu 
r_values = np.linspace(0, 5, 100)  
combined_pot_values = [combined_potential(r) for r in r_values]
plt.plot(r_values, combined_pot_values, label='Combined Potential')
plt.xlabel('r')
plt.ylabel('V(r)')
plt.title('Combined Potential')
plt.legend()
plt.grid(True)
plt.show()
