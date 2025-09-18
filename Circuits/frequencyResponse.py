import numpy as np
import matplotlib.pyplot as plt

def H(Z_1: float, Z_2: float) -> float:
    return Z_2/(Z_1 + Z_2)

c = 3e8 #m/s
wl = 30 # meters
f = c/wl
w = 2*np.pi*f
C = 1e-6 # Farads
Z_R = 1e3 #Ohms
Z_C = -1j/(w*C)
Hmag = np.sqrt(np.real(H(Z_R,Z_C))**2 + np.im(H(Z_R,Z_C))**2)
Hphase = np.arctan(np.im(H(Z_R,Z_C))/np.real(H(Z_R,Z_C)))
fc = 1/(2*np.pi*Z_R*C)


