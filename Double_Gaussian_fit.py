import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit


def gauss(x, mu, x0, sigma):
    return mu/(np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x - x0)**2/(2*sigma**2))

def double_gauss(x, mu1, x01, sigma1, mu2, x02, sigma2):
    gauss1 = gauss(x, mu1, x01, sigma1)
    gauss2 = gauss(x, mu2, x02, sigma2)
    return gauss1 + gauss2

def double_gauss_fit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(double_gauss, x, y, p0=[max(y), mean, sigma, max(y), mean, sigma])
    return popt

energy = []
counts = []

with open('Field_emission_distribution.csv', 'r') as csvfile:
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

mu1, x01, sigma1, mu2, x02, sigma2 = double_gauss_fit(energy, counts)
sigma1 = abs(sigma1)
sigma2 = abs(sigma2)

FWHM1 = 2*np.sqrt(2*np.log(2)) * sigma1
FWHM2 = 2*np.sqrt(2*np.log(2)) * sigma2


########## Plotting the curve ############
plt.scatter(energy, counts, s=5, c='k')
plt.plot(energy, double_gauss(energy, mu1, x01, sigma1, mu2, x02, sigma2), '--r', label='Double Gaussian fit')

plt.plot(energy, gauss(energy, mu1, x01, sigma1), 'b', label='First Gaussian')

plt.plot(energy, gauss(energy, mu2, x02, sigma2), 'g', label='Second Gaussian')

plt.xlim(-1.3, 1.8)
# naming the x axis
plt.xlabel('Energy (eV)')
# naming the y axis
plt.ylabel('Counts ($\\times 10^{4}$)')
# giving a title to my graph
plt.title('Field emission distribution')
plt.legend()

plt.show()


print('Weight of the 1st Gaussian: ' + str(round(mu1/(mu1+mu2), 5)))
print('Norminal energy of the 1st Gaussian: ' + str(round(x01, 5)) + ' eV')
print('Full width half maximum of the 1st Gaussian: ' +  str(round(FWHM1, 4)) + ' eV, corresponding to sigma1 = ' + str(round(sigma1, 4)))

print('Weight of the 2nd Gaussian: ' + str(round(mu2/(mu1+mu2), 5)))
print('Norminal energy of the 2nd Gaussian: ' + str(round(x02, 5)) + ' eV')
print('Full width half maximum of the 2nd Gaussian: ' +  str(round(FWHM2, 4)) + ' eV, corresponding to sigma2 = ' + str(round(sigma2, 4)))

