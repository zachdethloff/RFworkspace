import numpy as np
import matplotlib.pyplot as plt

def H(Z_1: float, Z_2: float) -> float:
    return Z_2/(Z_1 + Z_2)

def response_finder(fs: list,ftype: str, R: float=0, C: float=0,L: float=0) -> tuple[list[float],list[float]]:
    Hmags = []
    Hphases = []

    for f in fs:
        w = 2*np.pi*f
        Z_C = -1j/(w*C)
        if ftype == 'Low':
            Z_2 = Z_C
            Z_1 = R
            Hphase = -np.arctan(w*R*C)
            Hphases.append(Hphase * 180/np.pi)
        elif ftype == 'High':
            Z_2 = R
            Z_1 = Z_C
            Hphase = np.arctan(1/(w*R*C))
            Hphases.append(Hphase * 180/np.pi)
        elif ftype == 'Band':
            Z_1 = R + 1j*w*L
            Z_2 = 1/(1j*w*C)
            Hphase = np.arctan(1/(w*R*C))
            Hphases.append(Hphase * 180/np.pi)
        Hmag = np.sqrt(np.real(H(Z_1,Z_2))**2 + np.imag(H(Z_1,Z_2))**2)
        Hmags.append(Hmag)
    return Hmags, Hphases


c = 3e8 #m/s
wls = np.arange(300,1,-1) # meters
C = 1e-9 # Farads
R = 50 # Ohms
L = 1e-6 # H
fs = [c/wl for wl in wls] # Hz
fc = 1/(2*np.pi*R*C) # Hz
print("Frequency Cutoff at " + str(fc) + " Hz")

############################################ 

#  LOW PASS FILTER - RC

############################################

Hmags, Hphases = response_finder(fs,'Low',R,C)

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

Hmags, Hphases = response_finder(fs,'High',R,C)

plt.figure(figsize=(10,6))
plt.title("High Pass RC Filter Magnitude Response")
plt.plot([f/1e6 for f in fs],20*np.log10(np.abs(Hmags)))
plt.vlines(fc/1e6,20*np.log10(np.abs(min(Hmags))),20*np.log10(np.abs(max(Hmags))),colors='red', linestyles='solid')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude Response (dB)')
plt.savefig('HPmag.png', dpi=300, bbox_inches='tight')
plt.close()
print('Magnitude Response Plot Saved')

plt.figure(figsize=(10,6))
plt.title("High Pass RC Filter Phase Response")
plt.plot([f/1e6 for f in fs],Hphases)
plt.vlines(fc/1e6,min(Hphases),max(Hphases),colors='red', linestyles='solid')
plt.xlabel("Frequency (MHz)")
plt.ylabel('Phase Response (Deg)')
plt.savefig('HPphase.png', dpi=300, bbox_inches='tight')
plt.close()
print('Phase Response Plot Saved')

############################################

# BAND PASS FILTER - RLC

############################################

rez = 1/np.sqrt(L*C)
Q = rez*L/R
f_o = 1/(2*np.pi*np.sqrt(L*C))
print("Resonance at " + str(rez))
print("Quality Factor Q = " + str(Q))
print("Central Frequency = " + str(f_o))
Hmags, Hphases = response_finder(fs,'Band',R,C,L)

plt.figure(figsize=(10,6))
plt.title("Band Pass RLC Filter Magnitude Response")
plt.plot([f/1e6 for f in fs],20*np.log10(np.abs(Hmags)))
plt.vlines(fc/1e6,20*np.log10(np.abs(min(Hmags))),20*np.log10(np.abs(max(Hmags))),colors='red', linestyles='solid')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude Response (dB)')
plt.savefig('BPmag.png', dpi=300, bbox_inches='tight')
plt.close()
print('Magnitude Response Plot Saved')

plt.figure(figsize=(10,6))
plt.title("Band Pass RLC Filter Phase Response")
plt.plot([f/1e6 for f in fs],Hphases)
plt.vlines(fc/1e6,min(Hphases),max(Hphases),colors='red', linestyles='solid')
plt.xlabel("Frequency (MHz)")
plt.ylabel('Phase Response (Deg)')
plt.savefig('BPphase.png', dpi=300, bbox_inches='tight')
plt.close()
print('Phase Response Plot Saved')
