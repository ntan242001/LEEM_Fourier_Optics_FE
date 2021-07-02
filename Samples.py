import numpy as np
import matplotlib.pyplot as plt
import math


object_size = 400               # simulating object size in nm
simulating_steps = 1 + 2**15 # total simulating steps
# An array of points in the x space
x_array = (np.linspace(-object_size/2, object_size/2, simulating_steps) + object_size/simulating_steps)*1e-9

# A function to choose different sample object function
def create_object(object_type, k = 1, h=1):
    global object_function, object_amplitude, object_phase
    if object_type == "Step amplitude object":
    # Creating an 1:1/sqrt(2) step amplitude object whose phase is uniformly set to 0
        object_phase = np.zeros_like(x_array)

        object_amplitude = np.ones_like(x_array)

        for counter, element in enumerate(x_array):
            if element > 0:
                object_amplitude[counter] = 1/np.sqrt(2)

    if object_type == "Error function amplitude object":
    # Creating an error function amplitude object whose phase is uniformly set to 0         
        object_amplitude = np.ones_like(x_array)

        object_phase = np.zeros_like(x_array)

        for counter, element in enumerate(x_array):
            object_amplitude[counter] = math.erf(element*1e8)/2*(1-1/np.sqrt(2)) + (1+1/np.sqrt(2))/2

        object_amplitude = object_amplitude[::-1]

    if object_type == "Step phase object":
    # Creating a k.pi step phase object whose amplitude is uniformly set to 1        
        object_amplitude = np.ones_like(x_array)

        object_phase = np.zeros_like(x_array)

        for counter, element in enumerate(x_array):
            if element > 0:
                object_phase[counter] = k * np.pi


    if object_type == "Error function phase object":
    # Creating an error function phase object whose amplitude is uniformly set to 1
        object_amplitude = np.ones_like(x_array)

        object_phase = np.ones_like(x_array)

        for counter, element in enumerate(x_array):
            object_phase[counter] = (math.erf(element*h*1e8)+1)/2*k*np.pi

    # Object function
    object_function = np.multiply(object_amplitude, np.exp(1j * object_phase)) 

create_object("Error function phase object", k = 1)

##################################
##################################
'''
# Plotting the object
fig, ax1 = plt.subplots()
ax1.set_xlabel('object position (nm)', color = 'k')
ax1.set_ylabel('$ \\varphi $', color = 'k')
ax1.plot(x_array*1e9, object_phase, color = 'k')
ax1.tick_params(axis='y')
ax1.text(160, 2.9, 'Phase', fontsize=12)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('$\sigma$', color='tab:red')  # we already handled the x-label with ax1
ax2.plot(x_array*1e9, object_amplitude, color= 'tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.text(-180, 1.004, 'Amplitude', fontsize=12)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
'''
# Plotting the object
fig, ax1 = plt.subplots()
ax1.set_xlabel('object position (nm)', color = 'k')
ax1.set_ylabel('$ \\varphi $ (rad)', color = 'k')
ax1.tick_params(axis='y')
ax1.text(160, 2.9, 'Phase', fontsize=12)
ax2 = ax1.twinx()
ax2.set_ylabel('$\sigma$', color='tab:red')  # we already handled the x-label with ax1
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.text(-180, 1.05, 'Amplitude', fontsize=12)

#fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
for i in [0.1, 0.5, 1, 100]:
    create_object("Error function phase object", k = 1, h =i)   
    if i == 0.1:
        ax1.plot(x_array*1e9, object_phase, label = "scale $1 \\times 10^{7}$", color = 'b')
    if i == 0.5:
        ax1.plot(x_array*1e9, object_phase, label = "scale $5 \\times 10^{7}$", color = 'orange')
    if i == 1:
        ax1.plot(x_array*1e9, object_phase, label = "scale $1 \\times 10^{8}$", color = 'g')   
    if i == 100:
        ax1.plot(x_array*1e9, object_phase, label = "perfect step", color = 'k')
    
ax1.legend()    
ax2.plot(x_array*1e9, object_amplitude, "--",color= 'tab:red')    
ax2.set_ylim(0,3)

plt.show()
