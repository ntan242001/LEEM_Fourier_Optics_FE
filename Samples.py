import numpy as np
import matplotlib.pyplot as plt
import math


object_size = 400               # simulating object size in nm
simulating_steps = 1 + 2**15 # total simulating steps
# An array of points in the x space
x_array = (np.linspace(-object_size/2, object_size/2, simulating_steps) + object_size/simulating_steps)*1e-9

# A function to choose different sample object function
def create_object(object_type, k = 1):
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
            object_phase[counter] = (math.erf(element*1e8)+1)/2*k*np.pi

    # Object function
    object_function = np.multiply(object_amplitude, np.exp(1j * object_phase)) 

create_object("Error function phase object", k = 1)

##################################
##################################

# Plotting the object
plt.plot(x_array, object_amplitude)
plt.plot(x_array, object_phase)
