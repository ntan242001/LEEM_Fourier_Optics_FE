# Image simulation of pure amplitude object with varying aperture angle 

import numpy as np
import time

object_size = 400               # total simulating space in nm
simulation_steps = 1 + 2**10    # total simulating steps

# A function to choose different LEEM parameters
def choose_LEEM_type(constant_type):
    global E, E_0, C_c, C_cc, C_3c, C_3, C_5, alpha_ill, delta_E, M_L
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
        lamda_o = 6.6262e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * E_0)
        
        #q_max = alpha_ap / lamda
        q_ill = alpha_ill / lamda

choose_LEEM_type('IBM NAC')
print('LEEM chosen.')

# Creating a 1:1/sqrt(2) amplitude object whose phase is uniformly set to 0
simulated_space = np.linspace(-object_size/2, object_size/2, simulation_steps)

phase_shift = np.zeros_like(simulated_space)

amplitude = np.ones_like(simulated_space)
for counter, element in enumerate(simulated_space):
    if element > 0:
        amplitude[counter] = 1/np.sqrt(2)
        
# Object function
object_function = np.multiply(amplitude, np.exp(1j * phase_shift))
# The object image is reversed after passing through the lens
object_function = object_function[::-1]         















        

    
    