import numpy as np
import matplotlib.pyplot as plt
import csv

E = 15010  # eV  Nominal Energy After Acceleration
E_0 = 11.5 # eV  Energy at the sample ##########CUSTOMIZABLE INPUT##########
kappa = np.sqrt(E/E_0)

C_3 = 0.0297 * kappa**(1/2) + 0.1626  # m  Third Order Spherical Aberration Coefficient
C_5 = 0.6223 * kappa**(3/2) - 79.305  # m  Fifth Order Spherical Aberration Coefficient

lamda = 6.6261e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * E) # in metre

delta_z_series1 = []

resolution_list1 = []

with open('R(dz) Gaussian spread Step amplitude object nac_LEEM_E0=11.5.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row == []:
                continue
            else:
                delta_z_series1.append(float(row[0])*1e-6)
                resolution_list1.append(float(row[1]))    
    csvfile.close()
    
delta_z_series2 = []

resolution_list2 = []

with open('R(dz) FN spread Step amplitude object nac_LEEM_E0=11.5.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row == []:
                continue
            else:
                delta_z_series2.append(float(row[0])*1e-6)
                resolution_list2.append(float(row[1]))    
    csvfile.close()
    
delta_z_series3 = []
resolution_list3 = []

with open('R(dz) Gauss G1 spread Step amplitude object nac_LEEM_E0=11.5.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row == []:
                continue
            else:
                delta_z_series3.append(float(row[0])*1e-6)
                resolution_list3.append(float(row[1]))    
    csvfile.close()
  
delta_z_series4 = []
resolution_list4 = []

with open('R(dz) Gaussian spread Step amplitude object ac_LEEM_E0=11.5.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row == []:
                continue
            else:
                delta_z_series4.append(float(row[0])*1e-6)
                resolution_list4.append(float(row[1]))    
    csvfile.close()
    
    
delta_z_series5 = []
resolution_list5 = []

with open('R(dz) FN spread Step amplitude object ac_LEEM_E0=11.5.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row == []:
                continue
            else:
                delta_z_series5.append(float(row[0])*1e-6)
                resolution_list5.append(float(row[1]))    
    csvfile.close()
    
    
delta_z_series6 = []
resolution_list6 = []

with open('R(dz) Gauss G1 spread Step amplitude object ac_LEEM_E0=11.5.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row == []:
                continue
            else:
                delta_z_series6.append(float(row[0])*1e-6)
                resolution_list6.append(float(row[1]))    
    csvfile.close()
    
# plt.plot(delta_z_series1/(C_5*lamda**2)**(1/3), resolution_list1, label = 'nac Gaussian')
# plt.plot(delta_z_series2/(C_5*lamda**2)**(1/3), resolution_list2, label = 'nac FN (Triple Gaussian)')
# plt.plot(delta_z_series3/(C_5*lamda**2)**(1/3), resolution_list3, label = 'nac 1 of Triple Gaussian')

# plt.plot(delta_z_series4/(C_5*lamda**2)**(1/3), resolution_list4, label = 'ac Gaussian')
# plt.plot(delta_z_series5/(C_5*lamda**2)**(1/3), resolution_list5, label = 'ac FN (Triple Gaussian)')
# plt.plot(delta_z_series6/(C_5*lamda**2)**(1/3), resolution_list6, label = 'ac 1 of Triple Gaussian')

plt.plot(delta_z_series1/(C_3*lamda)**(1/2), resolution_list1, label = 'nac Gaussian')
plt.plot(delta_z_series2/(C_3*lamda)**(1/2), resolution_list2, label = 'nac FN (Triple Gaussian)')
plt.plot(delta_z_series3/(C_3*lamda)**(1/2), resolution_list3, label = 'nac 1 of Triple Gaussian')

plt.plot(delta_z_series4/(C_3*lamda)**(1/2), resolution_list4, label = 'ac Gaussian')
plt.plot(delta_z_series5/(C_3*lamda)**(1/2), resolution_list5, label = 'ac FN (Triple Gaussian)')
plt.plot(delta_z_series6/(C_3*lamda)**(1/2), resolution_list6, label = 'ac 1 of Triple Gaussian')



plt.xlim(-10,10)
plt.ylim(0,)

# naming the x axis
plt.xlabel('$\\frac{\Delta z}{(C_3 \lambda)^{1/2}}$', fontsize=18)
# naming the y axis
plt.ylabel('Resolution (nm)', fontsize=12)
plt.legend(bbox_to_anchor=(1, 1))

plt.show()
