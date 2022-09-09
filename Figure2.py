import numpy as np
import matplotlib.pyplot as plt

### Generating the Fowler-Nordheim distribution ### 
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
F = 4.5*1e9   # V/m

d_0_inv = 2*np.sqrt(2*phi*e*m)/(e*F*hbar) 

# An array of electron energy in eV 
Emin=4.6
Emax=6
E_array = np.linspace(Emin, Emax, 1000)

# Fermi-Dirac distribution
f_E = 1/(np.exp((E_array-E_F)*e/(k_B*T)) + 1)

counts = (np.exp(E_array*e*d_0_inv) - 1)*f_E

idx_max = np.argmax(counts)
counts_max = counts[idx_max]

counts_half = counts_max/2

left_index = np.argmin(np.abs(counts[0:idx_max] - counts_half))
right_index = idx_max + np.argmin(np.abs(counts[idx_max:-1] - counts_half))

G_FWHM = E_array[right_index] - E_array[left_index]

################################


N = 3

def gauss(x, A, x0, sigma):
    return A * np.exp(-((x - x0))**2/(2*sigma**2))

def N_gauss(N, x, A, x0, sigma):
    N_gauss = 0
    for i in range(N):
        N_gauss += gauss(x, A[i], x0[i], sigma[i])
    return N_gauss

A = [1.691*1e11, 4.031*1e11, 5.487*1e11]
x0 = [-0.3382, -0.1438, -0.03925]
c = [0.2815, 0.136, 0.07507]

sigma = c/np.sqrt(2)
FWHM = 2*np.sqrt(2*np.log(2)) * sigma

mu = []
for i in range(N):
    mu.append(A[i]*(np.sqrt(2*np.pi*sigma[i]**2)) )

mu_tot = sum(mu)
mu = mu/mu_tot

fig,ax=plt.subplots()
ax.axvline(x=0, linestyle = '--' ,color = 'k')

color = ['#f99a1c', (51/235, 204/235, 51/235), 'b']
for i in range(N):
    if i == 0:
        label = '$c_{1}(\epsilon)$'
    if i == 1:
        label = '$c_{2}(\epsilon)$'    
    if i == 2:
        label = '$c_{3}(\epsilon)$'
    ax.plot(E_array - E_F, gauss(E_array - E_F, A[i], x0[i], sigma[i])/counts_max, color = color[i], linewidth=2,linestyle='--', label=label)

########## Plotting the curve ############
ax.plot(E_array - E_F, counts/counts_max, 'k-', linewidth=2, label = 'FE')
ax.plot(E_array - E_F, N_gauss(N, E_array - E_F, A, x0, sigma)/counts_max, 'r-.', linewidth=2, label='triple-Gaussian')

s=15
ax.set_xlim(Emin-E_F, Emax-E_F)
ax.set_ylim(0, 1.1)
ax.minorticks_on()
ax.set_xlabel('Energy (eV)', fontsize=s)
ax.set_ylabel('Distribution', fontsize=s)
# plt.title('Field emission distribution')
# plt.text(-0.07, 0, '$E_F$')
ax.legend(frameon=False, fontsize=s)
fig.tight_layout()

for i in range(N):
    if i == 0:
        label = 'c_{1}(\epsilon)'
    if i == 1:
        label = 'c_{2}(\epsilon)'    
    if i == 2:
        label = 'c_{3}(\epsilon)'
    print(label)    
    print('Weight: ' + str(round(mu[i], 5)))
    print('Norminal energy: ' + str(round(x0[i], 5)) + ' eV')
    print('FWHM = ' +  str(round(FWHM[i], 4)) + ' eV; sigma = ' + str(round(sigma[i], 4)))
    print()
    
print('Single Gauss fit FWHM:', round(G_FWHM, 4))
#print(list(counts))


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