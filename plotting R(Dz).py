import numpy as np
import matplotlib.pyplot as plt
import csv

E = 15010  # eV  Nominal Energy After Acceleration
E_0 = 11.5 # eV  Energy at the sample ##########CUSTOMIZABLE INPUT##########
kappa = np.sqrt(E/E_0)

C_3 = 0.0297 * kappa**(1/2) + 0.1626  # m  Third Order Spherical Aberration Coefficient
C_5 = 0.6223 * kappa**(3/2) - 79.305  # m  Fifth Order Spherical Aberration Coefficient

lamda = 6.6261e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * E) # in metre

delta_z_series = []
resolution_list1 = []

with open('resolution_aperture_IBMnac.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row == []:
                continue
            else:
                delta_z_series.append(float(row[0]))
                resolution_list1.append(float(row[1]))    
    csvfile.close()
    
resolution_list2 = []

with open('resolution_aperture_IBMnac.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row == []:
                continue
            else:
                delta_z_series.append(float(row[0]))
                resolution_list2.append(float(row[1]))    
    csvfile.close()
    
resolution_list3 = []

with open('resolution_aperture_IBMnac.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row == []:
                continue
            else:
                delta_z_series.append(float(row[0]))
                resolution_list3.append(float(row[1]))    
    csvfile.close()
    
plt.plot(delta_z_series/(C_3*lamda)**(1/2), resolution_list1, label = 'Gaussian')
plt.plot(delta_z_series/(C_3*lamda)**(1/2), resolution_list2, label = 'FN (Triple Gaussian)')
plt.plot(delta_z_series/(C_3*lamda)**(1/2), resolution_list3, label = '1 of Triple Gaussian')

plt.xlim(-5.5,6)
plt.ylim(0,)

# naming the x axis
plt.xlabel('$\\frac{\Delta z}{(C_3 \lambda)^{1/2}}$', fontsize=18)
# naming the y axis
plt.ylabel('Resolution (nm)', fontsize=12)
plt.legend(bbox_to_anchor=(1, 1))

plt.show()

plt.plot(delta_z_series/(C_3*lamda)**(1/2), resolution_list1, label = 'Gaussian')
plt.plot(delta_z_series/(C_3*lamda)**(1/2), resolution_list2, label = 'FN (Triple Gaussian)')
plt.plot(delta_z_series/(C_3*lamda)**(1/2), resolution_list3, label = '1 of Triple Gaussian')

plt.xlim(-5.5,6)
plt.ylim(0,)

# naming the x axis
plt.xlabel('$\\frac{\Delta z}{(C_3 \lambda)^{1/2}}$', fontsize=18)
# naming the y axis
plt.ylabel('Resolution (nm)', fontsize=12)
plt.legend(bbox_to_anchor=(1, 1))

plt.show()

plt.plot(delta_z_series/(C_3*lamda)**(1/2), resolution_list1, label = 'Gaussian')
plt.plot(delta_z_series/(C_3*lamda)**(1/2), resolution_list2, label = 'FN (Triple Gaussian)')
plt.plot(delta_z_series/(C_3*lamda)**(1/2), resolution_list3, label = '1 of Triple Gaussian')

plt.xlim(-5.5,6)
plt.ylim(0,)

# naming the x axis
plt.xlabel('$\\frac{\Delta z}{(C_3 \lambda)^{1/2}}$', fontsize=18)
# naming the y axis
plt.ylabel('Resolution (nm)', fontsize=12)
plt.legend(bbox_to_anchor=(1, 1))

plt.show()
