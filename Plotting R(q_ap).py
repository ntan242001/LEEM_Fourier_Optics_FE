import matplotlib.pyplot as plt
import csv

alpha_ap_series = []
resolution_list = []

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
                alpha_ap_series.append(float(row[0]))
                resolution_list.append(float(row[1]))    
    csvfile.close()
    
plt.loglog(alpha_ap_series, resolution_list)

plt.title('Resolution versus aperture angle for IBM nac')
plt.xlabel('Aperture angle (mrad)') 
plt.ylabel('Resolution (nm)')

print(min(resolution_list))