import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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
        elif ftype == 'High':
            Z_2 = R
            Z_1 = Z_C
        elif ftype == 'Band':
            Z_1 = 1j*w*L + Z_C
            Z_2 = R
        Hphase = np.arctan(np.imag(H(Z_1,Z_2))/np.real(H(Z_1,Z_2)))
        Hphases.append(Hphase * 180/np.pi)
        Hmag = np.sqrt(np.real(H(Z_1,Z_2))**2 + np.imag(H(Z_1,Z_2))**2)
        Hmags.append(Hmag)
    return Hmags, Hphases


c = 3e8 #m/s
wls1 = np.arange(300,1,-1) # meters
wls2 = np.arange(2,1,-1/300)
wls = np.concatenate((wls1,wls2))
C = 1e-9 # Farads
R = 50 # Ohms
L = 1e-6 # H
fs = [c/wl for wl in wls] # Hz
fc = 1/(2*np.pi*R*C) # Hz
print("Frequency Cutoff at " + str(round(fc,-6)/1e6) + " MHz")

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

# BAND PASS/STOP FILTER - RLC

############################################

R = 10    # Ohms
L = 253e-9  # H
C = 10e-12 # F

rez = 1/(np.sqrt(L*C))
Q = rez*L/R
f_o = int(rez/(2*np.pi)/1e6)
bw = R/L
f_1 = (np.sqrt((R/(2*L))**2 + rez**2) - R/(2*L))/(2*np.pi)
f_2 = (np.sqrt((R/(2*L))**2 + rez**2) + R/(2*L))/(2*np.pi)
print(f"Quality Factor Q = {round(Q,2):.2f}")
print("Central Frequency = " + str(f_o) + " MHZ")
Hmags, Hphases = response_finder(fs,'Band',R,C,L)

plt.figure(figsize=(10,6))
plt.title("Band Pass RLC Filter Magnitude Response")
plt.plot([f/1e6 for f in fs],20*np.log10(np.abs(Hmags)))
plt.vlines(f_o,20*np.log10(np.abs(min(Hmags))),20*np.log10(np.abs(max(Hmags))),colors='red', linestyles='solid')
plt.hlines(20*np.log10(np.abs(max(Hmags))),f_1/1e6,f_2/1e6,colors='orange',linestyles='solid')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude Response (dB)')
plt.savefig('BPmag.png', dpi=300, bbox_inches='tight')
plt.close()
print('Magnitude Response Plot Saved')

plt.figure(figsize=(10,6))
plt.title("Band Stop RLC Filter Magnitude Response")
plt.plot([f/1e6 for f in fs],20*np.log10(np.abs([1-Hmag for Hmag in Hmags])))
plt.vlines(f_o,20*np.log10(np.abs(min(Hmags))),20*np.log10(np.abs(max(Hmags))),colors='red', linestyles='solid')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude Response (dB)')
plt.savefig('BSmag.png', dpi=300, bbox_inches='tight')
plt.close()
print('Magnitude Response Plot Saved')

plt.figure(figsize=(10,6))
plt.title("Band Pass RLC Filter Phase Response")
plt.plot([f/1e6 for f in fs],Hphases)
plt.vlines(f_o,min(Hphases),max(Hphases),colors='red', linestyles='solid')
plt.xlabel("Frequency (MHz)")
plt.ylabel('Phase Response (Deg)')
plt.savefig('BPphase.png', dpi=300, bbox_inches='tight')
plt.close()
print('Phase Response Plot Saved')

###############################################