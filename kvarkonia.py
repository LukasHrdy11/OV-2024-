import numpy as np

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
    return 2*reduk_hm/hbarc**2 * (E - potencial(r) - hbarc**2 * l * (l + 1) / (2 * reduk_hm * r**2))

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
                #print(b)
            else:
                a = midpoint
                #print(a)
            iter_count += 1
            if iter_count > max_iter:
                return None
            
        return (a + b) / 2

    result = bisection_method(numerov, Emin, Emax)
    return result


def find_bound_states(E_ranges, potencial, l, h):
    energies = []
    for Emin, Emax in E_ranges:
        energy = bound_energy(Emin, Emax, potencial, l, h)
        if energy is not None:
            energies.append(energy)
    return energies


# Constants
h = 0.01  # fm
rmax = 30  # fm
rgrid = np.linspace(0,30,num=int(rmax/h+1),endpoint=True) #grid of r starting from 0 to rmax (step h)
l = 0
a = 787  # MeV/fm
b = 137.8  # MeV
c = 1200  # MeV^-1
mc = 1900 # MeV
mb = 5250  # MeV
hbarc = 197.32697 #MeV*fm
reduk_hm = mc * mc / (mc + mc)  # 1/(MeV fm**2)
gamma = 0.5772
Lambda = 500  # MeV
mu = (Lambda * np.exp(gamma))**-1  # Výpočet pro μ
E_ranges = [(0, 1000),(1000, 2000), (2000, 3000), (3000, 4000), (5000, 6000), (6000, 7000)] #MeV


bound_states = find_bound_states(E_ranges, potencial, l, h)
print(f"Nalezené vazebné energie: {bound_states}")