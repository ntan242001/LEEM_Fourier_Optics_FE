import matplotlib.pyplot as plt
import csv

x_array = []
matrixI_step = []
matrixI_108 = []
matrixI_5107 = []
matrixI_107 = []

with open('Step phase object IBM_nac.csv', 'r') as csvfile:
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
                matrixI_step.append(float(row[1]))    
    csvfile.close()

with open('10^8 Error function phase object IBM_nac.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row == []:
                continue
            else:
                matrixI_108.append(float(row[1]))    
    csvfile.close()
    
with open('5*10^7 Error function phase object IBM_nac.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row == []:
                continue
            else:
                matrixI_5107.append(float(row[1]))    
    csvfile.close()    

with open('10^7 Error function phase object IBM_nac.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row == []:
                continue
            else:
                matrixI_107.append(float(row[1]))    
    csvfile.close() 
    
########## Plotting the curve ############
#plt.subplot(111)
plt.plot(x_array, matrixI_step, label = "perfect step")
plt.plot(x_array, matrixI_108, label = "scale $10^8$")
plt.plot(x_array, matrixI_108, label = "scale $5 \\times 10^7$")
plt.plot(x_array, matrixI_108, label = "scale $10^7$")

plt.xlim(-10, 10)
# naming the x axis
plt.xlabel('Position x (nm)')
# naming the y axis
plt.ylabel('Instensity')
# giving a title to my graph
plt.title('$\pi$ step phase object')
plt.legend()

plt.show()

