import matplotlib.pyplot as plt
import csv

x_array = []
matrixI_nac = []
matrixI_ac = []

with open('Step phase objectI(x)_IBM_nac.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row == []:
                continue
            else:
                x_array.append(float(row[0]))
                matrixI_nac.append(float(row[1]))    
    csvfile.close()
    
with open('Step phase objectI(x)_IBM_ac.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row == []:
                continue
            else:
                matrixI_ac.append(float(row[1]))    
    csvfile.close()    

    
########## Plotting the curve ############
#plt.subplot(111)
plt.plot(x_array, matrixI_nac, label = "nac")
plt.plot(x_array, matrixI_ac, label = "ac")

plt.xlim(-10, 10)
# naming the x axis
plt.xlabel('Position x (nm)')
# naming the y axis
plt.ylabel('Instensity')
# giving a title to my graph
plt.title('$\pi$ step phase object')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.show()
