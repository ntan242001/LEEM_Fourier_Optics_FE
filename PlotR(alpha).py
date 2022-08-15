import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.signal import savgol_filter

a_array_nac = []
R_G1_nac = []
R_FN_nac = []

a_array_ac = []
R_G1_ac = []
R_FN_ac = []


with open('Step amplitude object_R(a)_G1_nac.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row == []:
                continue
            else:
                a_array_nac.append(float(row[0]))
                R_G1_nac.append(float(row[1]))   
    csvfile.close()
    
a_array_nac = np.array(a_array_nac)*1e3

with open('Step amplitude object_R(a)_FN_nac.csv', 'r') as csvfile:
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
    
with open('Step amplitude object_R(a)_G1_ac.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row == []:
                continue
            else:
                a_array_ac.append(float(row[0]))
                R_G1_ac.append(float(row[1]))   
    csvfile.close()
    
a_array_ac = np.array(a_array_ac)*1e3

with open('Step amplitude object_R(a)_FN_ac.csv', 'r') as csvfile:
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
    
smooth_R_G1_nac = savgol_filter(R_G1_nac, 31, 8)
smooth_R_FN_nac = savgol_filter(R_FN_nac, 31, 8)
smooth_R_G1_ac = savgol_filter(R_G1_ac, 31, 9)
smooth_R_FN_ac = savgol_filter(R_FN_ac, 31, 9)
    
########## Plotting the curves ############
fig, ax = plt.subplots()

ax.loglog(a_array_ac, smooth_R_G1_ac, 'm-', label = "AC, Gaussian")
ax.loglog(a_array_ac, smooth_R_FN_ac, 'c-', label = "AC, FN")
ax.loglog(a_array_nac, smooth_R_G1_nac, 'r-', label = "NAC, Gaussian")
ax.loglog(a_array_nac, smooth_R_FN_nac, 'b-', label = "NAC, FN")

import matplotlib.ticker as ticker
def myLogFormat(y,pos):
    # Find the number of decimal places required
    decimalplaces = int(np.maximum(-np.log10(y),0))     # =0 for numbers >=1
    # Insert that number into a format string
    formatstring = '{{:.{:1d}f}}'.format(decimalplaces)
    # Return the formatted tick label
    return formatstring.format(y)

ax.yaxis.set_major_formatter(ticker.FuncFormatter(myLogFormat))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(myLogFormat))


yminimum, ymaximum = 0.4, 20

ax.axvline(x=2.34, linestyle = '--', c = 'r')
ax.axvline(x=1.95, linestyle = '--', c = 'b')
ax.axvline(x=7.37, linestyle = '--', c = 'm')
ax.axvline(x=7.03, linestyle = '--', c = 'c')

ax.set_xlabel("Aperture angle (mrad)")
ax.set_ylabel("Resolution (nm)")
ax.set_ylim(yminimum, ymaximum)
ax.set_title("Resolution versus aperture angle")
ax.legend()
