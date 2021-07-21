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

'''
General model Gauss3:
     f(x) = 
              a1*exp(-((x-b1)/c1)^2) + a2*exp(-((x-b2)/c2)^2) + 
              a3*exp(-((x-b3)/c3)^2)
Coefficients (with 95% confidence bounds):
       a1 =   5.487e+11  (3.847e+11, 7.127e+11)
       b1 =    -0.03925  (-0.04167, -0.03684)
       c1 =     0.07507  (0.06672, 0.08342)
       a2 =   1.691e+11  (1.329e+11, 2.052e+11)
       b2 =     -0.3382  (-0.4119, -0.2646)
       c2 =      0.2815  (0.2384, 0.3247)
       a3 =   4.031e+11  (2.726e+11, 5.336e+11)
       b3 =     -0.1438  (-0.1771, -0.1104)
       c3 =       0.136  (0.1019, 0.1701)

Goodness of fit:
  SSE: 1.176e+22
  R-square: 0.9979
  Adjusted R-square: 0.9978
  RMSE: 1.137e+10
'''
