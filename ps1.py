import math

# Otevření souboru
with open('phase_shifts.txt', 'r') as file:
    # Načtení řádků ze souboru a rozdělení na sloupce
    lines = file.readlines()[1:]  # přeskočení prvního řádku
    data = [line.strip().split('\t') for line in lines]

# Vytvoření seznamu s daty
data_list = []

# Převod řetězce na číslo pro každý sloupec a přidání do seznamu
for row in data:
    energy = float(row[0])
    phaseshift = float(row[1])
    data_list.append([energy, phaseshift])

# Vypsání načtených dat
#print(data_list)


energy_values = []

# Přidání hodnot z prvního sloupce do nového seznamu
for row in data:
    energy = float(row[0])
    energy_values.append(energy)

#print(energy_values)


def potencial(r):
        return 200*pow(math.e, -1.487*pow(r,2)) - 178*pow(math.e, -0.639*pow(r, 2))



def phases(E):
    my = 938/2
    hbarc = 197.32697
    k = math.sqrt(2*my*E)/hbarc
    rmax = 30
    h = 0.01
    j0 = math.sin(k*(rmax-2*h))/(k*(rmax-2*h))
    n0 = -math.cos(k*(rmax-2*h))/(k*(rmax-2*h))
    j1 = math.sin(k*(rmax-h))/(k*(rmax-h))
    n1 = -math.cos(k*(rmax-h))/(k*(rmax-h))
    l = 0

    def numerov(E, potencial,l):
            
            u0 = 0
            u1 = h**(l+1)
            x = int(rmax/h)
            u = []
            u2 = 0
            for i in range (0,x-2):
                k0 = 2*my/hbarc**2*(E - potencial(0+i*h)-hbarc**2*l*(l+1)/(2*my*rmax**2))
                k1 = 2*my/hbarc**2*(E - potencial(h+i*h)-hbarc**2*l*(l+1)/(2*my*rmax**2))
                k2 = 2*my/hbarc**2*(E - potencial(2*h+i*h)-hbarc**2*l*(l+1)/(2*my*rmax**2))
                u2 = (2*(1-5/12*pow(h,2)*k1)*u1-(1+1/12*pow(h,2)*k0)*u0)/(1+pow(h,2)/12*k2)
                u0 = u1
                u1 = u2
                
            return u0,u1
    u1, u2 = numerov(E,potencial,l)



    delta_l = math.atan((k*(rmax-h)*j0-u1/u2*k*j1)/(n0+u1/u2*n1))*180/math.pi
    if delta_l < 0:
        delta_l = delta_l+180
    delta_l
    print(delta_l)

for energy in energy_values:
    phases(energy)