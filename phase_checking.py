import numpy as np
import math

rmax = 30
h = 0.01
hbarc = 197.32697
l = 0

def potential(r):
    return 200 * np.exp(-1.487 * np.power(r, 2)) - 178 * np.exp( -0.639 * np.power(r, 2))

my_phase = 938/2
my_num = 938.272 * 939.565 / (938.272 + 939.565)

energy_for_phase = [0.03260052732991577, 0.618570920141414, 2.809174938470539, 9.654077810188886, 19.101650821694978, 41.71985774394201, 63.480841901287825, 91.51080273814753, 119.52411262722937, 159.63028382419736, 165.5480757789632]
energy_for_numerov = np.zeros(len(energy_for_phase))

for i, energy in enumerate(energy_for_phase):
     energy_for_numerov[i] = 2*energy_for_phase[i]

def numerov(E, potencial, rmax, l, h):
        hbarc = 197.32697
        u0 = 0
        h=0.01
        u1 = h ** (l + 1)
        x = int(rmax / h)
        u = np.zeros(x)
        u[0] = u0
        u[1] = u1

        for i in range(0, x - 2):
            k0 = 2 * my_num / hbarc ** 2 * (E - potencial(0 + i * h) - hbarc ** 2 * l * (l + 1) / (2 * my_num * rmax ** 2))
            k1 = 2 * my_num / hbarc ** 2 * (E - potencial(h + i * h) - hbarc ** 2 * l * (l + 1) / (2 * my_num * rmax ** 2))
            k2 = 2 * my_num / hbarc ** 2 * (E - potencial(2 * h + i * h) - hbarc ** 2 * l * (l + 1) / (2 * my_num * rmax ** 2))
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
        return u_over_r[x-2], u_over_r[x-1]

def calculate_phase_shift(energy_phase, energy_num, potential):
    k = np.sqrt(2 * my_phase * energy_phase) / hbarc 
    j0 = np.sin(k * (rmax - 2 * h)) / (k * (rmax - 2 * h))
    n0 = -np.cos(k * (rmax - 2 * h)) / (k * (rmax - 2 * h))
    j1 = np.sin(k * (rmax - h)) / (k * (rmax - h))
    n1 = -np.cos(k * (rmax - h)) / (k * (rmax - h))
    u0, u1 = numerov(energy_num, potential,rmax,h,l)
    delta_l = np.degrees(np.arctan2((u1/u0) * (np.sqrt(2 * my_phase * energy_phase) / hbarc) * (j0 * (rmax - h) - j1 * (rmax - 2 * h)),(n0 * (rmax - h) - n1 * (rmax - 2 * h))))
    if delta_l < 0:
        delta_l += 180
    return delta_l

for i,energy in enumerate(energy_for_phase):
     print("Fázový posun pro energii:", energy_for_phase[i],"s energií numerova: ", energy_for_numerov[i],"je: ",calculate_phase_shift(energy_for_phase[i],energy_for_numerov[i],potential))

