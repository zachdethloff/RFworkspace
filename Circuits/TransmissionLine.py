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
z = np.arange(0,5,100)
wl = 0.3

for Zl in Load:
    vrange = []
    for x in z:
        rc = (Zl - Zo)/(Zl + Zo)
        vmag = np.abs(V(x,wl,rc))
        vrange.append(vmag)
        


