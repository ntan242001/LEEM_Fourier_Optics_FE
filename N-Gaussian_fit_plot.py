import numpy as np
import matplotlib.pyplot as plt
import csv

N = 2

def gauss(x, A, x0, sigma):
    return A * np.exp(-(x - x0)**2/(2*sigma**2))

def N_gauss(N, x, A, x0, sigma):
    N_gauss = 0
    for i in range(N):
        N_gauss += gauss(x, A[i], x0[i], sigma[i])
    return N_gauss

energy = []
counts = []

with open('AM45 FE distribution.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row == []:
                continue
            else:
                energy.append(float(row[0]))
                counts.append(float(row[1]))    
    csvfile.close()

energy = np.array(energy)
counts = np.array(counts)
energyticks = np.linspace(energy[0], energy[-1], 100)

A = [128.5, 49.51]
x0 = [0.02246, 0.2099]
c = [0.2117, 0.3888]

sigma = c/np.sqrt(2)
FWHM = 2*np.sqrt(2*np.log(2)) * sigma

mu = []
for i in range(N):
    mu.append(A[i]*(np.sqrt(2*np.pi*sigma[i]**2)) )

mu_tot = sum(mu)
mu = mu/mu_tot


########## Plotting the curve ############
plt.scatter(energy, counts, s=5, c='k')
plt.plot(energyticks, N_gauss(N, energyticks, A, x0, sigma), 'r', label='Double Gaussian fit')

color = ['b', 'g', 'o', 'p']
for i in range(N):
    if i == 0:
        label = '1st Gaussian'
    if i == 1:
        label = '2nd Gaussian'    
    if i == 2:
        label = '3rd Gaussian'
    if i == 4:
        label = '4th Gaussian'
    plt.plot(energyticks, gauss(energyticks, A[i], x0[i], sigma[i]), color = color[i], label=label)


#plt.xlim(-1.3, 1.8)
#plt.ylim(0, 175)
# naming the x axis
plt.xlabel('Energy (eV)')
# naming the y axis
plt.ylabel('Counts')
# giving a title to my graph
plt.title('Field emission distribution')
plt.legend()

plt.show()

for i in range(N):
    if i == 0:
        label = '1st Gaussian'
    if i == 1:
        label = '2nd Gaussian'    
    if i == 2:
        label = '3rd Gaussian'
    if i == 4:
        label = '4th Gaussian'
    print(label)    
    print('Weight: ' + str(round(mu[i], 5)))
    print('Norminal energy: ' + str(round(x0[i], 5)) + ' eV')
    print('FWHM = ' +  str(round(FWHM[i], 4)) + ' eV; sigma = ' + str(round(sigma[i], 4)))
    print()
