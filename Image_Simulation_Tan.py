# Image simulation of pure amplitude object with varying aperture angle 

import numpy as np
import time

object_size = 400               # simulating object size in nm
simulating_steps = 1 + 2**10    # total simulating steps

# A function to choose different LEEM parameters
def choose_LEEM_type(constant_type):
    global E, E_0, C_c, C_cc, C_3c, C_3, C_5, alpha_ill, delta_E, M_L, lamda, lamda_0, q_ill
    if constant_type == "IBM AC":
        E = 15010  # eV  Nominal Energy After Acceleration
        E_0 = 10  # eV  Starting Energy
        C_c = 0  # m   Second Rank Chromatic Aberration Coefficient
        C_cc = 27.9 # m   Third Rank Chromatic Aberration Coefficient
        C_3c = -67.4 # m   Forth Rank Chromatic Aberration Coefficient
        C_3 = 0  # m   Spherical Aberration Coefficient
        C_5 = 92.8
    
        #alpha_ap = 7.37e-3  # rad Acceptance Angle of the Contrast Aperture
        alpha_ill = 0.1e-3  # rad Illumination Divergence Angle
    
        delta_E = 0.25  # eV  Energy Spread
        M_L = 0.653  # Lateral Magnification
    
    if constant_type =='IBM NAC':
        
        E = 15010  # eV  Nominal Energy After Acceleration
        E_0 = 10  # eV  Starting Energy
        C_c = -0.075  # m   Chromatic Aberration Coefficient
        C_cc = 23.09 # m   Third Rank Chromatic Aberration Coefficient
        C_3c = -59.37  # m   Forth Rank Chromatic Aberration Coefficient
        C_3 = 0.345  # m  Third Order Spherical Aberration Coefficient
        C_5 = 39.4  # m  Fifth Order Spherical Aberration Coefficient
        
        #alpha_ap = 2.34e-3  # rad Acceptance Angle of the Contrast Aperture
        alpha_ill = 0.1e-3  # rad Illumination Divergence Angle
        
        delta_E = 0.25  # eV  Energy Spread
        M_L = 0.653  # Lateral Magnification
        
    lamda = 6.6262e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * E)
    lamda_0 = 6.6262e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * E_0)
    
    #q_max = alpha_ap / lamda
    q_ill = alpha_ill / lamda
        
        
# A function to set different defocus values
def choose_defocus(defocus_type):
    global delta_z
    if defocus_type == "in-focus":
        delta_z = 0
    elif defocus_type == "Scherzer defocus":
        delta_z = np.sqrt(3/2*C_3*lamda)
    elif defocus_type == "A Phi Scherzer defocus":
        delta_z = np.sqrt(9/64*C_5*lamda**2)
    

choose_LEEM_type('IBM NAC')
print('LEEM chosen.')

# Creating a 1:1/sqrt(2) amplitude object whose phase is uniformly set to 0
# An array of points in the x space
simulated_space = np.linspace(-object_size/2, object_size/2, simulating_steps)

object_phase = np.zeros_like(simulated_space)

object_amplitude = np.ones_like(simulated_space)
for counter, element in enumerate(simulated_space):
    if element > 0:
        object_amplitude[counter] = 1/np.sqrt(2)
        
# Object function
object_function = np.multiply(object_amplitude, np.exp(1j * object_phase))
# The object image is reversed after passing through the lens
object_function = object_function[::-1]         

# The Fourier Transform of the Object Wave Function
F_object_function = np.fft.fftshift(np.fft.fft(object_function, simulating_steps) * (1 / simulating_steps))

# An array of points in the q space
q = 1 / (object_size * 1e-9) * np.arange(0, simulating_steps, 1)
q = q - (np.max(q) - np.min(q)) / 2












































        

    
    