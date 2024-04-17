import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from scipy.integrate import quad
from scipy.misc import derivative
from mpl_toolkits.mplot3d import Axes3D

def potencial(r):
    return 200 * pow(math.e, -1.487 * pow(r, 2)) - 178 * pow(math.e, -0.639 * pow(r, 2))

def second_derivative(f, x):
    return derivative(f, x, n=2)

def fc(rmax, h, potencial, l):
    my = 938.272 * 939.565 / (938.272 + 939.565)
    Emin = -6
    Emax = 0
    hbarc = 197.32697

    def function(E, potencial):
        u0 = 0
        u1 = h ** (l + 1)
        x = int(rmax / h)
        u2 = 0
        for i in range(0, x - 2):
            k0 = 2 * my / hbarc ** 2 * (E - potencial(0 + i * h) - hbarc ** 2 * l * (l + 1) / (2 * my * rmax ** 2))
            k1 = 2 * my / hbarc ** 2 * (E - potencial(h + i * h) - hbarc ** 2 * l * (l + 1) / (2 * my * rmax ** 2))
            k2 = 2 * my / hbarc ** 2 * (E - potencial(2 * h + i * h) - hbarc ** 2 * l * (l + 1) / (2 * my * rmax ** 2))
            u2 = (2 * (1 - 5 / 12 * pow(h, 2) * k1) * u1 - (1 + 1 / 12 * pow(h, 2) * k0) * u0) / (1 + pow(h, 2) / 12 * k2)
            u0 = u1
            u1 = u2
        return u2

    def bisection_method(f, a, b, tol=1e-5, max_iter=30):
        iter_count = 0
        while (b - a) / 2 > tol:
            midpoint = (a + b) / 2
            if f(midpoint, potencial) == 0:
                return midpoint
            elif f(a, potencial) * f(midpoint, potencial) < 0:
                b = midpoint
            else:
                a = midpoint
            iter_count += 1
            if iter_count > max_iter:
                return None

        return (a + b) / 2

    a = Emin
    b = Emax

    result = bisection_method(function, a, b)
    return result

def najit_ustalene_x(energie, tolerance):
    ustalene_x = []
    ustaleni_x = None

    for i in range(1, len(energie)):
        change = abs(energie[i] - energie[i - 1])
        if change <= tolerance and ustaleni_x is None:
            ustaleni_x = i
        elif change > tolerance and ustaleni_x is not None:
            ustalene_x.append(ustaleni_x)
            ustaleni_x = None

    if ustaleni_x is not None:
        ustalene_x.append(ustaleni_x)

    return ustalene_x

def pouziti_fc1():
    def function2(E, potencial, rmax, l, h):
        my = 938.272 * 939.565 / (938.272 + 939.565)
        hbarc = 197.32697
        u0 = 0
        u1 = h ** (l + 1)
        x = int(rmax / h)
        u = np.zeros(x)
        u[0] = u0
        u[1] = u1

        for i in range(0, x - 2):
            k0 = 2 * my / hbarc ** 2 * (E - potencial(0 + i * h) - hbarc ** 2 * l * (l + 1) / (2 * my * rmax ** 2))
            k1 = 2 * my / hbarc ** 2 * (E - potencial(h + i * h) - hbarc ** 2 * l * (l + 1) / (2 * my * rmax ** 2))
            k2 = 2 * my / hbarc ** 2 * (E - potencial(2 * h + i * h) - hbarc ** 2 * l * (l + 1) / (2 * my * rmax ** 2))
            u2 = (2 * (1 - 5 / 12 * pow(h, 2) * k1) * u1 - (1 + 1 / 12 * pow(h, 2) * k0) * u0) / (
                        1 + pow(h, 2) / 12 * k2)
            u[i + 2] = u2
            u0 = u1
            u1 = u2

        r = np.arange(0, rmax, h)
        u_over_r = np.zeros_like(u)
        for i in range(len(r)):
            if r[i] == 0:
                u_over_r[i] = 0
            else:
                u_over_r[i] = u[i] / r[i] * (1 / (4 * math.pi) ** 0.5)
        return u_over_r
        
         

    def probability_function_squared(x, y, z, tck):
        r = np.sqrt(x**2 + y**2 + z**2)
        if r >= rmax:
            return 0
        else:
            return np.abs(splev(r, tck))**2 * 10

    def plot_probability_surface(tck, r_final):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = r_final * np.outer(np.cos(u), np.sin(v))
        y = r_final * np.outer(np.sin(u), np.sin(v))
        z = r_final * np.outer(np.ones(np.size(u)), np.cos(v))

        
        prob_values = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                prob_values[i, j] = probability_function_squared(x[i, j], y[i, j], z[i, j], tck)

        
        surf = ax.plot_surface(x, y, z, cmap='viridis', facecolors=plt.cm.viridis(prob_values),
                               linewidth=0, antialiased=False)

        ax.set_title('Probability Density Surface')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    h1 = 0.1
    h2 = 0.01
    h3 = 0.001
    l = 0
    xs = []
    ys1 = []
    ys2 = []
    ys3 = []

    ustalene_E = []
    tolerance = 0.01  

    for i in range(30):
        vysledek = fc(i, h1, potencial, l)
        xs.append(i)
        ys1.append(vysledek)

    ustalene_x = najit_ustalene_x(ys1, tolerance)

    for i in range(1, len(ys1)):
        change = abs(ys1[i] - ys1[i - 1])
        if change <= tolerance:
            ustalene_E.append(ys1[i])

    for z in range(30):
        vysledek = fc(z, h2, potencial, l)
        ys2.append(vysledek)

    for w in range(30):
        vysledek = fc(w, h3, potencial, l)
        ys3.append(vysledek)

    E_final = min(ustalene_E)
    r_final = max(ustalene_x)

    print(r_final)

    rmax = r_final
    l = 0
    h = 0.01

    r = np.arange(0, rmax, h)
    
    
    u_over_r = function2(E_final, potencial, rmax, l, h)
    tck = splrep(r, u_over_r, s=0)  

   
    integral, error = quad(lambda x: np.conjugate(splev(x, tck)) * potencial(x) * splev(x, tck) * x**2, 0, rmax)

    integral1, error1 = quad(lambda x: np.conjugate(splev(x, tck)) * x**2 * splev(x, tck)* x**2, 0, rmax)

    my = 938.272 * 939.565 / (938.272 + 939.565)
    hbarc = 197.32697

    integral2, error2 = quad(lambda x: -(hbarc)**2/2*my * np.conjugate(splev(x, tck)) * second_derivative(lambda x: splev(x, tck), x) * x**2, 0, rmax)

    print("Exeption values:")

    print("Potenciál:", integral)
    print("chyba:", error)

    print("r**2:", integral1)
    print("chyba:", error1)

    print("Kinetická enrgie:", integral2)
    print("chyba:", error2)

    plt.figure(figsize=(8, 6))
    plt.plot(r, u_over_r, color='b', linestyle='-', linewidth=1)
    plt.title('První potenciál')
    plt.xlabel('r [fermi]')
    plt.ylabel('Wave function')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    
    plot_probability_surface(tck, 2)

pouziti_fc1()
