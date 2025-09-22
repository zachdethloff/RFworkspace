import numpy as np
import matplotlib.pyplot as plt

def H(Z_1: float, Z_2: float) -> float:
    return Z_2/(Z_1 + Z_2)

c = 3e8 #m/s
wls = np.arange(300,1,-1) # meters
C = 1e-9 # Farads
Z_R = 50 #Ohms

############################################ 

#  LOW PASS FILTER - RC

############################################

fs = [c/wl for wl in wls]
Hmags = []
Hphases = []
fc = 1/(2*np.pi*Z_R*C)
print("Frequency Cutoff at " + str(fc) + " Hz")

for f in fs:
    w = 2*np.pi*f
    Z_C = -1j/(w*C)
    Hmag = np.sqrt(np.real(H(Z_R,Z_C))**2 + np.imag(H(Z_R,Z_C))**2)
    Hmags.append(Hmag)
    Hphase = -np.arctan(w*Z_R*C)
    Hphases.append(Hphase * 180/np.pi)

plt.figure(figsize=(10,6))
plt.title("Low Pass RC Filter Magnitude Response")
plt.plot([f/1e6 for f in fs],20*np.log10(np.abs(Hmags)))
plt.vlines(fc/1e6,20*np.log10(np.abs(min(Hmags))),20*np.log10(np.abs(max(Hmags))),colors='red', linestyles='solid')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude Response (dB)')
plt.savefig('LPmag.png', dpi=300, bbox_inches='tight')
plt.close()
print('Magnitude Response Plot Saved')

plt.figure(figsize=(10,6))
plt.title("Low Pass RC Filter Phase Response")
plt.plot([f/1e6 for f in fs],Hphases)
plt.vlines(fc/1e6,min(Hphases),max(Hphases),colors='red', linestyles='solid')
plt.xlabel("Frequency (MHz)")
plt.ylabel('Phase Response (Deg)')
plt.savefig('LPphase.png', dpi=300, bbox_inches='tight')
plt.close()
print('Phase Response Plot Saved')


############################################

# HIGH PASS FILTER - CR

############################################



