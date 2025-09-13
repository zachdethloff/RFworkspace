import numpy as np
import matplotlib.pyplot as plt

# no = 376.73
# b = outer conductor radius
# a = inner conductor radius

# Zo = no/(2*np.pi) * np.log(b/a) - Characteristic Impedence

#alpha = attenuation constant
# P(x) = Po*np.exp(-2*alpha*x)

def V(z: int, wl: float, rc: float):
    return 1 + rc*np.exp(-2j * 2*np.pi*z/wl)

Zo = 50
Load = [25,50,75,100]
z = np.arange(0,5,1/100)
wl = 0.3
plt.figure(figsize=(10,6))

for Zl in Load:
    vrange = []
    for x in z:
        rc = (Zl - Zo)/(Zl + Zo)
        vmag = np.abs(V(x,wl,rc))
        vrange.append(vmag)
    print("VSWR for Impedance Load of " + str(Zl) + " Ohms = " + str((1+np.abs(rc))/(1-np.abs(rc))) + "\n")
    plt.plot(z,vrange,label=str(Zl) + " Ohms")

plt.legend()
plt.title("Voltage Down A Transmission Line With Characteristic Impedance of 50 Ohms")
plt.xlabel("Distance (m)")
plt.ylabel("Voltage (V)")
plt.savefig('my_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print('Plot has been saved')


        


