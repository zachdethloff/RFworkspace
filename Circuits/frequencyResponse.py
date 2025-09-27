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
            ctf = H(Z_1,Z_2)
        elif ftype == 'High':
            Z_2 = R
            Z_1 = Z_C
            ctf = H(Z_1,Z_2)
        elif ftype == 'Band':
            Z_1 = 1j*w*L + Z_C
            Z_2 = R
            ctf = H(Z_1,Z_2)
        elif ftype =='Tank':
            Z_1 = 1j*w*L + Z_C
            Z_2 = R
            ctf = 1 - H(Z_1,Z_2)
        elif ftype == 'CF':
            Z_1 = 1j*w*L
            Z_2 = R/(1+1j*w*R*C)
            ctf = H(Z_1,Z_2)
        Hphase = np.angle(ctf)
        Hphases.append(Hphase * 180/np.pi)
        Hmag = np.abs(ctf)
        Hmags.append(Hmag)
    return Hmags, Hphases


c = 3e8 #m/s
C = 1e-9 # Farads
R = 50 # Ohms
L = 1e-6 # H
fs = [val*1e6 for val in np.arange(1,300,1)] # Hz
Mfs = [f/1e6 for f in fs] #MHz
fc = 1/(2*np.pi*R*C)/1e6 # Hz
print(f"Frequency Cutoff at {fc:.1f} MHz")

############################################ 

#  LOW PASS FILTER - RC

############################################

Hmags, Hphases = response_finder(fs,'Low',R,C)

plt.figure(figsize=(10,6))
plt.title("Low Pass RC Filter Magnitude Response")
plt.plot(Mfs,20*np.log10(Hmags))
plt.vlines(fc,20*np.log10(min(Hmags)),20*np.log10(max(Hmags)),colors='red', linestyles='solid')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude Response (dB)')
plt.savefig('LPmag.png', dpi=300, bbox_inches='tight')
plt.close()
print('Magnitude Response Plot Saved')

plt.figure(figsize=(10,6))
plt.title("Low Pass RC Filter Phase Response")
plt.plot(Mfs,Hphases)
plt.vlines(fc,min(Hphases),max(Hphases),colors='red', linestyles='solid')
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
plt.plot(Mfs,20*np.log10(Hmags))
plt.vlines(fc,20*np.log10(min(Hmags)),20*np.log10(max(Hmags)),colors='red', linestyles='solid')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude Response (dB)')
plt.savefig('HPmag.png', dpi=300, bbox_inches='tight')
plt.close()
print('Magnitude Response Plot Saved')

plt.figure(figsize=(10,6))
plt.title("High Pass RC Filter Phase Response")
plt.plot(Mfs,Hphases)
plt.vlines(fc,min(Hphases),max(Hphases),colors='red', linestyles='solid')
plt.xlabel("Frequency (MHz)")
plt.ylabel('Phase Response (Deg)')
plt.savefig('HPphase.png', dpi=300, bbox_inches='tight')
plt.close()
print('Phase Response Plot Saved')

############################################

# BAND PASS FILTER - RLC SERIES

############################################

R = 10    # Ohms
L = 253e-9  # H
C = 10e-12 # F

rez = 1/(np.sqrt(L*C))
Q = rez*L/R
f_o = int(rez/(2*np.pi)/1e6)
bw = R/L
f_1 = (np.sqrt((R/(2*L))**2 + rez**2) - R/(2*L))/(2*np.pi)/1e6
f_2 = (np.sqrt((R/(2*L))**2 + rez**2) + R/(2*L))/(2*np.pi)/1e6
print(f"Quality Factor Q = {round(Q,2):.2f}")
print("Central Frequency = " + str(f_o) + " MHZ")
Hmags, Hphases = response_finder(fs,'Band',R,C,L)

plt.figure(figsize=(10,6))
plt.title("Band Pass RLC Filter Magnitude Response")
plt.plot(Mfs,20*np.log10(Hmags))
plt.vlines(f_o,20*np.log10(min(Hmags)),20*np.log10(max(Hmags)),colors='red', linestyles='solid')
plt.hlines(20*np.log10(max(Hmags)),f_1,f_2,colors='orange',linestyles='solid')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude Response (dB)')
plt.savefig('BPmag.png', dpi=300, bbox_inches='tight')
plt.close()
print('Magnitude Response Plot Saved')

plt.figure(figsize=(10,6))
plt.title("Band Pass RLC Filter Phase Response")
plt.plot(Mfs,Hphases)
plt.vlines(f_o,min(Hphases),max(Hphases),colors='red', linestyles='solid')
plt.xlabel("Frequency (MHz)")
plt.ylabel('Phase Response (Deg)')
plt.savefig('BPphase.png', dpi=300, bbox_inches='tight')
plt.close()
print('Phase Response Plot Saved')

###############################################

# NOTCH FILTER - TANK 

###############################################

Hmags, Hphases = response_finder(fs,'Tank',R,C,L)

plt.figure(figsize=(10,6))
plt.title("Band Stop RLC Filter Magnitude Response")
plt.plot(Mfs,20*np.log10(Hmags))
plt.vlines(f_o,20*np.log10(min(Hmags)),20*np.log10(max(Hmags)),colors='red', linestyles='solid')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude Response (dB)')
plt.savefig('BSmag.png', dpi=300, bbox_inches='tight')
plt.close()
print('Magnitude Response Plot Saved')

plt.figure(figsize=(10,6))
plt.title("Band Stop RLC Filter Phase Response")
plt.plot(Mfs,Hphases)
plt.vlines(f_o,min(Hphases),max(Hphases),colors='red', linestyles='solid')
plt.xlabel("Frequency (MHz)")
plt.ylabel('Phase Response (Deg)')
plt.savefig('BSphase.png', dpi=300, bbox_inches='tight')
plt.close()
print('Phase Response Plot Saved')

###############################################

# CHEBYSHEV FILTER - 2ND ORDER

###############################################

L = 56e-9
C = 68e-12
R = 50

Hmags, Hphases = response_finder(fs,'CF',R,C,L)

fc = 1/(2*np.pi*np.sqrt(L*C))/1e6 # MHz

plt.figure(figsize=(10,6))
plt.title("Chebyshev Filter Magnitude Response")
plt.plot(Mfs,20*np.log10(Hmags))
plt.vlines(fc,20*np.log10(min(Hmags)),20*np.log10(max(Hmags)),colors='red', linestyles='solid')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude Response (dB)')
plt.savefig('CFmag.png', dpi=300, bbox_inches='tight')
plt.close()
print('Magnitude Response Plot Saved')

plt.figure(figsize=(10,6))
plt.title("Chebysev Filter Phase Response")
plt.plot(Mfs,Hphases)
plt.vlines(fc,min(Hphases),max(Hphases),colors='red', linestyles='solid')
plt.xlabel("Frequency (MHz)")
plt.ylabel('Phase Response (Deg)')
plt.savefig('CFphase.png', dpi=300, bbox_inches='tight')
plt.close()
print('Phase Response Plot Saved')