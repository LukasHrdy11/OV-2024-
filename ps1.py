import math


with open('phase_shifts.txt', 'r') as file:
    
    lines = file.readlines()[1:]  
    data = [line.strip().split('\t') for line in lines]


data_list = []


for row in data:
    energy = float(row[0])
    phaseshift = float(row[1])
    data_list.append([energy, phaseshift])


#print(data_list)


energy_values = []


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
    r2 = rmax - h
    r1 = rmax - 2*h
    j1 = math.sin(k * r1) / (k*r1)
    n1 = -math.cos(k * r1) / (k * r1)
    j2 = math.sin(k * r2) / (k*r2)
    n2 = -math.cos(k * r2) / (k * r2)
    l = 0

    def numerov(E, potencial,l):
            
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
                
            return u0,u1
    
    u1, u2 = numerov(E,potencial,l)
    b = u1/u2 * r2/r1
    up = j1- b*j2
    down =n1- b*n2

    delta_l = math.atan(up/down) * 180/(math.pi)
    if delta_l < 0:
        delta_l = delta_l+180
    delta_l
    print(delta_l)


energies1 = [0.5, 2.5, 5.0, 12.5, 25.0, 50.0, 75.0, 100.0]
energies2 = [0.03260052732991577, 0.618570920141414, 2.809174938470539, 9.654077810188886, 19.101650821694978, 41.71985774394201, 63.480841901287825, 91.51080273814753, 119.52411262722937, 159.63028382419736, 165.5480757789632]
energies = [0.0000000001, 10, 20, 30, 50, 73, 100, 163, 180]

for energy in energy_values:
    phases(energy)