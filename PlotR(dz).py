import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.signal import savgol_filter

C_3 = 0.345 
C_5 = 92.8

E = 15010 
lamda = 6.6261e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * E)

delta_z_series_nac = []
R_G1_nac = []
R_FN_nac = []

delta_z_series_ac = []
R_G1_ac = []
R_FN_ac = []

with open('Run5_R(dz)_G1_nac.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row == []:
                continue
            else:
                delta_z_series_nac.append(float(row[0]))
                R_G1_nac.append(float(row[1]))    
    csvfile.close()

with open('Run5_R(dz)_FN_nac.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row == []:
                continue
            else:
                R_FN_nac.append(float(row[1]))    
    csvfile.close()
    
with open('R(dz) G1 ac_LEEM.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row == []:
                continue
            else:
                delta_z_series_ac.append(float(row[0]))
                R_G1_ac.append(float(row[1]))    
    csvfile.close()

with open('R(dz) FN ac_LEEM.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row == []:
                continue
            else:
                R_FN_ac.append(float(row[1]))    
    csvfile.close()
    

    
delta_z_seriesGauss = np.array(delta_z_series_nac)
delta_z_seriesFN = np.array(delta_z_series_ac)


smooth_R_G1_nac = savgol_filter(R_G1_nac, 3, 2)
smooth_R_FN_nac = savgol_filter(R_FN_nac, 3, 2)
smooth_R_G1_ac = savgol_filter(R_G1_ac, 33, 9)
smooth_R_FN_ac = savgol_filter(R_FN_ac, 33, 9)

########## Plotting the curves ############
fig, ax = plt.subplots()

width = 2
ax.plot(delta_z_series_nac/((C_3*lamda)**(1/2)), smooth_R_G1_nac, 'm-', linewidth=width, label = "NAC, Gaussian")
ax.plot(delta_z_series_nac/((C_3*lamda)**(1/2)), smooth_R_FN_nac, 'c-', linewidth=width, label = "NAC, FN")
# ax.plot(delta_z_series_ac/((C_5*lamda**2)**(1/3)), smooth_R_G1_ac, 'r-', linewidth=width, label = "AC, Gaussian")
# ax.plot(delta_z_series_ac/((C_5*lamda**2)**(1/3)), smooth_R_FN_ac, 'b-', linewidth=width, label = "AC, FN")

# ax.set_ylim(0, 7)
ax.set_xlim(-1.8,1.8)

ax.set_ylabel("Resolution (nm)")
ax.set_xlabel('$\Delta z ((C_3 \lambda)^{1/2}, (C_5 \lambda^2)^{1/3})$')
ax.set_xticks([-1,0,1])
ax.set_yticks([0,2,4,6])
ax.minorticks_on()
ax.legend()

ax.axvline(x=6.90e-7/((C_3*lamda)**(1/2)), linestyle = '--', c = 'm', linewidth=2)
ax.axvline(x=1.37e-6/((C_3*lamda)**(1/2)), linestyle = '--', c = 'c')
# ax.axvline(x=4.73e-8/((C_5*lamda**2)**(1/3)), linestyle = '--', c = 'r')
# ax.axvline(x=7.1e-8/((C_5*lamda**2)**(1/3)), linestyle = '--', c = 'b')


fig.tight_layout()

# naming the x axis
# plt.text(-0.8,-1,'$\Delta z ( $', color='k', fontsize=15)
# plt.text(-0.5,-1,'$(C_3 \lambda)^{1/2} $', color='b', fontsize=13)
# plt.text(0.05,-1,', ', color='k', fontsize=15)
# plt.text(0.12,-1,'$(C_5 \lambda^2)^{1/3} $', color='r', fontsize=13)
# plt.text(0.8,-1,')', color='k', fontsize=15)
# plt.xlabel('Defocus current (mA)')
