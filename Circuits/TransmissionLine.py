import numpy as np
import matplotlib.pyplot as plt

# no = 376.73
# b = outer conductor radius
# a = inner conductor radius

# Zo = no/(2*np.pi) * np.log(b/a) - Characteristic Impedence

#alpha = attenuation constant
# P(x) = Po*np.exp(-2*alpha*x)

def V(z: int, wl: float, rc: float) -> float:
    return 1 + rc*np.exp(-2j * 2*np.pi*z/wl)

def rc(Zl: float, Zo: int) -> float:
    return (Zl - Zo)/(Zl + Zo)

c = 3e8
Zo = 50
Load = [25,50,75,100]
z = np.arange(0,5,1/100)
wl = 0.3
f = c/wl
plt.figure(figsize=(10,6))

for Zl in Load:
    vrange = []
    for x in z:
        vmag = np.real(V(x,wl,rc(Zl,Zo)))
        vrange.append(vmag)
    print("VSWR for Impedance Load of " + str(Zl) + " Ohms = " + str((1+np.abs(rc(Zl,Zo)))/(1-np.abs(rc(Zl,Zo)))) + "\n")
    plt.plot(z,vrange,label=str(Zl) + " Ohms")

plt.legend()
plt.title("Voltage Down A Transmission Line With Characteristic Impedance of 50 Ohms")
plt.xlabel("Distance (m)")
plt.ylabel("Voltage (V)")
plt.savefig('my_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print('Plot has been saved')


#####################################################################

# Now to do this for specific components to model a wave coming through a transmission line and in to a pulse amplifier

plt.figure(figsize=(10,6))

R = 50 # Ohm Resistor
C = 5e-12 # F Capacitor
L = 5e-9 # H Inductor
wls = [300,30,3,0.3,0.1]

for wl in wls:
    f = c/wl
    Z_L = R + 1/(2j*np.pi*f*C) + 2j*np.pi*f*L #  Input impedance of the PA
    vrange =[]
    for x in z:
        vreal = np.real(V(x,wl,rc(Z_L,Zo)))
        vrange.append(vreal)
    print("Load for " + str(f/1e9) + "GHz: " + str(Z_L))
    plt.plot(z,vrange,label=str(f)+"Hz")
    #print("VSWR for Impedance Load of " + str(Z_L) + " Ohms = " + str((1+np.abs(rc(Z_L,Zo)))/(1-np.abs(rc(Z_L,Zo)))) + "\n")
plt.legend()
plt.title("Voltage Down A Transmission Line With Frequency Dependent Impedance")
plt.xlabel("Distance (m)")
plt.ylabel("Voltage (V)")
plt.savefig('my_plot_2.png', dpi=300, bbox_inches='tight')
plt.close()
print('Plot has been saved')

        


