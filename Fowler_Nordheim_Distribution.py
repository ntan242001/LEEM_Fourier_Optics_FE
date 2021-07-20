import numpy as np
import matplotlib.pyplot as plt

k_B = 1.38*1e-23
hbar = 1.055*1e-34
m = 9.1*1e-31
e = 1.6*1e-19  # C     
# The work function for tungsten
phi = 4.5    # eV
# Fermi level of tungsten
E_F = 5.77   # eV
# Temperature
T = 300      # K
# Electric field
F = 4.5*1e9

d_0_inv = 2*np.sqrt(2*phi*e*m)/(e*F*hbar) 

# An array of electron energy in eV 
E_array = np.linspace(4.6, 6, 100)

# Fermi-Dirac distribution
f_E = 1/(np.exp((E_array-E_F)*e/(k_B*T)) + 1)

counts = (np.exp(E_array*e*d_0_inv) - 1)*f_E

plt.plot(E_array - E_F, counts)
plt.axvline(x=0, color = 'k')

plt.ylabel('Counts')
plt.xlabel('Electron energy (eV)')

plt.text(-0.07, 0, '$E_F$')
print(list(counts))
