import numpy as np
from scipy import constants as c
from scipy.integrate import simps
from multiprocessing import Pool
import matplotlib.pyplot as plt


me = c.value("electron mass energy equivalent in MeV")
mp = c.value("proton mass energy equivalent in MeV")
mn = c.value("neutron mass energy equivalent in MeV")
h_c = 197.3269804  
m_deuterium = mn * mp / (mp + mn)


def V_c(r):
    x = r * 0.7
    return np.array((-10.463 * np.exp(-x) + 105.468 * np.exp(-2 * x) - 3187.8 * np.exp(-4 * x) + 9924.3 * np.exp(-6 * x)) / x)

def V_T(r):
    x = r * 0.7
    return np.array((-10.463 * ((1 + 3/x + 3/(x**2)) * np.exp(-x) - (12/x + 3/(x**2)) * np.exp(-4 * x)) + 351.77 * np.exp(-4 * x) - 1673.5 * np.exp(-6 * x)) / x)

def V_LS(r):
    x = r * 0.7
    return np.array((708.91 * np.exp(-4 * x) - 2713.1 * np.exp(-6 * x)) / x)


def U(r, mu, r_cutoff=100):
    A = np.zeros((2, 2))
    C1 = 2 * mu / (h_c ** 2)
    if r < r_cutoff:
        A[0, 0] = V_c(r)
        A[0, 1] = A[1, 0] = np.sqrt(8) * V_T(r)
        A[1, 1] = V_c(r) + 6 / (C1 * r**2) - 3 * V_LS(r) - 2 * V_T(r)
    return np.matrix(A)



def k_n(R, E, mu):
    C1 = 2 * mu / (h_c ** 2)
    kn = [np.matrix(C1 * (E * np.identity(2) - U(r, mu))) for r in R]
    return kn


def numerov_matrix(y_initial, k, h, back=False):
    y_0, y_1 = y_initial
    y_solution = [y_0, y_1]
    N = len(k)
    
    if back:
        k = k[::-1]

    for n in range(1, N-1):
        y_2 = y_n(k, y_0, y_1, h, n)
        y_0 = y_1
        y_1 = y_2
        y_solution.append(y_2)
    
    if back:
        return y_solution[::-1]
    return y_solution

def y_n(k, y_0, y_1, h, n):
    denominator = np.linalg.inv((np.identity(2) + h**2 / 12 * k[n+1]))
    nominator_1 = 2 * (np.identity(2) - 5 * h**2 / 12 * k[n]) * y_1
    nominator_2 = (np.identity(2) + h**2 / 12 * k[n-1]) * y_0
    return denominator * (nominator_1 - nominator_2)


def dy_n(y, k, h):
    N = len(y) - 1
    dy = [1 / (2 * h) * ((np.identity(2) + h**2 / 6 * k[n+1]) * y[n+1] - (np.identity(2) + h**2 / 6 * k[n-1]) * y[n-1]) for n in range(1, N)]
    return dy

def determinant_E_rc(y_in, y_out, dy_in, dy_out):
    U_tilde = dy_out[0] - np.linalg.inv(y_in[1]).T * dy_in[-1] * y_out[-2]
    detE = np.min(np.linalg.eigvals(U_tilde))
    return detE




def energy_determinants(r, E_min, E_max, dE, mu, rc_ratio=4/7):
    h = np.abs(r[1] - r[0])
    y_0 = initial(h=h, mode="out")
    rc_index = int(np.shape(r)[0] * rc_ratio)
    determinants = []

    for E in np.arange(E_min, E_max + dE, dE):
        y_in0 = initial(E, r[-1], h, mode="in")
        k = k_n(r, E, mu)
        y_out = numerov_matrix(y_0, k[:rc_index + 2], h)
        y_in = numerov_matrix(y_in0, k[rc_index - 1:], h, True)
        dy_out = dy_n(y_out[-3:], k[rc_index - 1:rc_index + 2], h)
        dy_in = dy_n(y_in[:3], k[rc_index - 1:rc_index + 2], h)
        detE = determinant_E_rc(y_in, y_out, dy_in, dy_out)
        determinants.append(detE)
    return determinants




def bound_energy(determinants, E_min, dE):
    value = np.min(determinants)
    i = np.where(np.isclose(value, determinants, atol=0))[0][0]
    return E_min + dE * i





def wavefunction(R, y_in_initial, y_out_initial, E_bound, mu, rc_ratio=4/7):
    rc_index = int(np.shape(R)[0] * rc_ratio)
    k = k_n(R, E_bound, mu)
    y_out = numerov_matrix(y_out_initial, k[:rc_index + 2], h)
    y_in = numerov_matrix(y_in_initial, k[rc_index - 1:], h, True)
    dy_out = dy_n(y_out[-3:], k[rc_index - 1:rc_index + 2], h)
    dy_in = dy_n(y_in[:3], k[rc_index - 1:rc_index + 2], h)
    U_tilde = dy_out[0] - np.linalg.inv(y_in[1]).T * dy_in[0] * y_out[-2]
    c_out = np.array([[1], [-U_tilde[1, 0] / U_tilde[1, 1]]])
    c_in = (np.linalg.inv(y_in[1]) * y_out[-2]).dot(c_out)
    y_bound_out = np.array([(y_out[i]).dot(c_out) for i in range(len(y_out) - 1)])
    y_bound_in = np.array([(y_in[i]).dot(c_in) for i in range(2, len(y_in))])
    y_full = np.concatenate((y_bound_out, y_bound_in))
    return y_full




def initial(E=1, r_max=1, h=1e-3, m=m_deuterium, mode="both"):
    y_0_out = np.matrix([[(0)**(0+1), 0], [0, (0)**(1+1)]])
    y_1_out = np.matrix([[(2*h)**(0+1), 0], [0, (2*h)**(1+1)]])
    y_out_initial = [y_0_out, y_1_out]

    alpha = np.sqrt(-2 * m * E) / h_c
    y_0_in = np.matrix([[np.exp(-alpha * r_max), 0], [0, (1 + 3 / (alpha * r_max) + 3 / (alpha * r_max)**2) * np.exp(-alpha * r_max)]])
    y_1_in = np.matrix([[np.exp(-alpha * (r_max - h)), 0], [0, (1 + 3 / (alpha * (r_max - h)) + 3 / (alpha * (r_max - h))**2) * np.exp(-alpha * (r_max - h))]])
    y_in_initial = [y_0_in, y_1_in]

    if mode == "both":
        return [y_in_initial, y_out_initial]
    elif mode == "in":
        return y_in_initial
    elif mode == "out":
        return y_out_initial



def normalize(y_bound, R):
    y_bound_components = [y_bound[:, i, 0] for i in range(0, np.shape(y_bound)[1])]
    C = np.sqrt(sum([simps(np.abs(y)**2, R) for y in y_bound_components]))
    y_bound_normalized = [y / C for y in y_bound_components]
    return y_bound_normalized




E_min, E_max, dE = -20, -0.1, 0.01
r_max = 14
h = 1e-1
r = np.arange(h, r_max, h)
rc = 7 / r_max

dets_E = np.array(energy_determinants(r, E_min, E_max, dE, m_deuterium, rc))
E_bound = bound_energy(np.abs(dets_E), E_min, dE)
print("Bound state energy:", E_bound, "MeV")

y_in0, y_out0 = initial(E_bound, r_max, h)
y_bounds = wavefunction(r, y_in0, y_out0, E_bound, m_deuterium, rc)
y_component = normalize(y_bounds, r)
u = y_component[0]
w = y_component[1]

np.savetxt("deuterium_solution.txt", np.array([r, y_component[0], y_component[1]]).T, delimiter=" ;\t ", header="r [fm] \t\t\t u(r) \t\t\t\t w(r) ")


plt.xlim(right=14)  
plt.ylim(top=0.55)
plt.xlim(left=0)  
plt.ylim(bottom=0)
plt.plot(r, y_component[0], label=r"s-vlna")
plt.plot(r, y_component[1], label=r"d-vlna")
plt.xlabel("r [fm]", fontsize=18)
plt.ylabel("[a.u.]", fontsize=18)
plt.legend(fontsize=15)
plt.show()


