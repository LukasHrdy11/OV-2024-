import numpy as np
import matplotlib.pyplot as plt

# Define potential and related functions
def alpha_s(r):
    return (12 * np.pi) / (25 * (2 * np.log(mu / r)))

def V_AF(r):
    return -(4 / 3) * (alpha_s(r) / r)

def V_int(r):
    return -b * np.exp(-r / c)

def V0(r):
    return V_AF(r) + a * r

def potencial(r):
    return V0(r) + V_int(r)

def k_function(E, potencial, r, l):
    return 2 * reduk_hm / hbarc**2 * (E - potencial(r) - hbarc**2 * l * (l + 1) / (2 * reduk_hm * r**2))

def bound_energy(Emin, Emax, potencial, l, h):
    def numerov(E, potencial, l, h):
        u0 = 0
        u1 = h ** (l + 1)
        k0 = k_function(E, potencial, rgrid[1], l)
        k1 = k_function(E, potencial, rgrid[2], l)
        
        for i in range(3, len(rgrid)):
            k2 = k_function(E, potencial, rgrid[i], l)
            u2 = (2 * (1 - 5 / 12 * h**2 * k1) * u1 - (1 + 1 / 12 * h**2 * k0) * u0) / (1 + h**2 / 12 * k2)

            k0 = k1
            k1 = k2

            u0 = u1
            u1 = u2
            
        return u2 

    def bisection_method(f, a, b, tol=1e-5, max_iter=30):
        iter_count = 0
        while (b - a) / 2 > tol:
            midpoint = (a + b) / 2
            if f(midpoint, potencial, l, h) == 0:
                return midpoint
            elif f(a, potencial, l, h) * f(midpoint, potencial, l, h) < 0:
                b = midpoint
            else:
                a = midpoint
            iter_count += 1
            if iter_count > max_iter:
                return None
        return (a + b) / 2

    result = bisection_method(numerov, Emin, Emax)
    return result


def find_bound_states_for_l(l, inicial_Emin, inicial_Emax, energy_step, num_bound_states=3):
    energies = []
    Emin, Emax = inicial_Emin, inicial_Emax
    while len(energies) < num_bound_states:
        energy = bound_energy(Emin, Emax, potencial, l, h)
        if energy is not None and energy < Emax - 1e-2:  
            energies.append(energy)
            Emin, Emax = Emax + energy_step, Emax + 2 * energy_step  
        else:
            Emin, Emax = Emax, Emax + energy_step  
    return energies


# Constants
h = 0.01  # fm
rmax = 30  # fm
rgrid = np.linspace(0, 30, num=int(rmax / h + 1), endpoint=True) 
a = 787  # MeV/fm
b = 137.8  # MeV
c = 1200  # MeV^-1
mc = 1900  # MeV
mb = 5250  # MeV
hbarc = 197.32697  # MeV*fm
reduk_hm = mc * mc / (mc + mc)  # 1/(MeV fm**2)
gamma = 0.5772
Lambda = 500  # MeV
mu = (Lambda * np.exp(gamma))**-1 

# Define initial values for each l
initial_values = {
    0: {'inicial_Emin': 3000, 'inicial_Emax': 3250, 'energy_step': 250},
    1: {'inicial_Emin': 3400, 'inicial_Emax': 3550, 'energy_step': 150},
    2: {'inicial_Emin': 3700, 'inicial_Emax': 3900, 'energy_step': 200}
}

# Find bound states for each l
all_bound_states = {}
for l, params in initial_values.items():
    all_bound_states[l] = find_bound_states_for_l(
        l, params['inicial_Emin'], params['inicial_Emax'], params['energy_step']
    )

# Print bound states
for l, energies in all_bound_states.items():
    print(f"Bound states for l={l}: {energies}")



experimental_data = {
    0: [3096, 3686, 4030],  # l=0 (S states)
    1: [3521],              # l=1 (P states)
    2: [3770, 4159]         # l=2 (D states)
}


calculated_data = {
    0: all_bound_states[0],  # l=0 (S states)
    1: all_bound_states[1],  # l=1 (P states)
    2: all_bound_states[2]   # l=2 (D states)
}


l_labels = {0: "S", 1: "P", 2: "D"}


plt.figure(figsize=(8, 6))

for l, energies in calculated_data.items():
    for energy in energies:
        plt.hlines(y=energy, xmin=l, xmax=l + 0.2, color="red", linewidth=2, label="Vypočet " if l == 0 and energy == energies[0] else "")


for l, energies in experimental_data.items():
    for energy in energies:
        plt.hlines(y=energy, xmin=l - 0.2, xmax=l, color="green", linewidth=2, label="Experiment" if l == 0 and energy == energies[0] else "")

plt.ylabel("E (MeV)", fontsize=20)
plt.xlabel(r"$c\bar{c}$ spektrum", fontsize=20)
plt.xticks([0, 1, 2], [l_labels[l] for l in [0, 1, 2]])  
plt.ylim(3000, 4700)


legend = plt.legend(loc="lower right", frameon=False, handletextpad=0)
legend.get_texts()[0].set_color("red")     
legend.get_texts()[1].set_color("green")    


for handle in legend.legendHandles:
    handle.set_visible(False)

plt.show()