import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time
import csv
import math

##############################
######### Preamble ###########
##############################

t_0 = time.time()

# A function to choose different LEEM parameters
def choose_LEEM_type(LEEM_type_str, aberration_corrected = False):
    global E, E_0, C_c, C_cc, C_3c, C_3, C_5, alpha_ap, alpha_ill, \
        delta_E, M_L, lamda, lamda_0, q_ill, q_ap, LEEM_type, E_0_series
    LEEM_type = LEEM_type_str
    
    if LEEM_type == "IBM":
        if aberration_corrected == False:
            E = 15010  # eV  Nominal Energy After Acceleration
            E_0 = 10  # eV  Energy at the sample
            
            C_c = -0.075  # m  Second Rank Chromatic Aberration Coefficient
            C_cc = 23.09 # m   Third Rank Chromatic Aberration Coefficient
            C_3c = -59.37  # m   Forth Rank Chromatic Aberration Coefficient
            
            C_3 = 0.345  # m  Third Order Spherical Aberration Coefficient
            C_5 = 39.4  # m  Fifth Order Spherical Aberration Coefficient
            
            alpha_ap = 2.34e-3  # rad Aperture angle
            alpha_ill = 0.1e-3  # rad Illumination Divergence Angle
            
            delta_E = 0.25  # eV  Energy Spread
            M_L = 0.653  # Lateral Magnification
            print("IBM NAC chosen.")
            
        elif aberration_corrected == True:
            E = 15010  # eV  Nominal Energy After Acceleration
            E_0 = 10  # eV  Energy at the sample
            
            C_c = 0  # m   Second Rank Chromatic Aberration Coefficient
            C_cc = 27.9 # m   Third Rank Chromatic Aberration Coefficient
            C_3c = -67.4 # m   Forth Rank Chromatic Aberration Coefficient
            
            C_3 = 0  # m   Spherical Aberration Coefficient
            C_5 = 92.8
        
            alpha_ap = 7.37e-3  # rad Aperture angle
            alpha_ill = 0.1e-3  # rad Illumination Divergence Angle
        
            delta_E = 0.25  # eV  Energy Spread
            M_L = 0.653  # Lateral Magnification
            print("IBM AC chosen.")
            
        lamda = 6.6261e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * E)
        lamda_0 = 6.6261e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * E_0)
    
        q_ap = alpha_ap/lamda
        q_ill = alpha_ill/lamda
        
    elif LEEM_type == "Energy dependent":
        if aberration_corrected == False:
            E = 15010  # eV  Nominal Energy After Acceleration
            E_0 = 20 # eV  Energy at the sample ##########CUSTOMIZABLE INPUT##########
            kappa = np.sqrt(E/E_0)
            
            C_c = -0.0121 * kappa**(1/2) + 0.0029 # m  Second Rank Chromatic Aberration Coefficient
            C_cc = 0.5918 * kappa**(3/2) - 87.063 # m   Third Rank Chromatic Aberration Coefficient
            C_3c = -1.2141 * kappa**(3/2) + 169.41  # m   Forth Rank Chromatic Aberration Coefficient
            
            C_3 = 0.0297 * kappa**(1/2) + 0.1626  # m  Third Order Spherical Aberration Coefficient
            C_5 = 0.6223 * kappa**(3/2) - 79.305  # m  Fifth Order Spherical Aberration Coefficient
            
            delta_E = 0.25  # eV  Energy Spread
            alpha_ill = 0.1e-3  # rad Illumination divergence angle
            M_L = 0.653  # Lateral Magnification
            print("Custom nac LEEM at " + str(E_0) + "eV chosen.")
            
        if aberration_corrected == True:
            E = 15010  # eV  Nominal Energy After Acceleration
            E_0 = 20 # eV  Energy at the sample
            kappa = np.sqrt(E/E_0)
            
            C_c = 0 # m  Second Rank Chromatic Aberration Coefficient
            C_cc = 0.5984 * kappa**(3/2) - 84.002 # m   Third Rank Chromatic Aberration Coefficient 
            C_3c = -1.1652 * kappa**(3/2) + 153.58  # m   Forth Rank Chromatic Aberration Coefficient  
            
            C_3 = 0  # m  Third Order Spherical Aberration Coefficient
            C_5 = 0.5624 * kappa**(3/2) - 16.541  # m  Fifth Order Spherical Aberration Coefficient
            
            delta_E = 0.25  # eV  Energy Spread
            alpha_ill = 0.1e-3  # rad Illumination divergence angle
            M_L = 0.653  # Lateral Magnification
            print("Custom nac LEEM at " + str(E_0) + "eV chosen.")   
        
        alpha_ap = (lamda/C_3)**(1/4) # rad Aperture angle for optimal resolution
        
        lamda = 6.6261e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * E) # in metre
        lamda_0 = 6.6261e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * E_0) # in metre
        
        q_ap = alpha_ap/lamda
        q_ill = alpha_ill/lamda

choose_LEEM_type("IBM", aberration_corrected = False)

# A function to set different defocus values
def choose_defocus(defocus_type):
    global delta_z
    if defocus_type == "In-focus":
        delta_z = 0
        print("In-focus chosen.")
    elif defocus_type == "Scherzer defocus":
        delta_z = np.sqrt(3/2*C_3*lamda)
        print("Scherzer defocus chosen.")
    elif defocus_type == "A-Phi Scherzer defocus":
        delta_z = np.sqrt(9/64*C_5*lamda**2)
        print("A-Phi Scherzer defocus chosen.")

choose_defocus("In-focus")


object_size = 400               # simulating object size in nm
simulating_steps = 1 + 2**15    # total simulating steps
# An array of points in the x space
x_array = (np.linspace(-object_size/2, object_size/2, simulating_steps) + object_size/simulating_steps)*1e-9


# A function to choose different sample object function
def create_object(object_type_str, k = 1):
    global object_type, object_function, object_amplitude, object_phase
    object_type = object_type_str
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

create_object("Step phase object", k = 1)

##################################
######## End of Preamble #########
##################################



##################################
########### Main Part ############
##################################
print("Simulation start.")
# The object image is reversed through the lens
object_function_reversed = object_function[::-1] 

# Initialising the series of function I(x) at different values of E_0
matrixI = np.zeros_like(x_array, dtype=complex)


# The Fourier Transform of the Object Wave Function
F_object_function = np.fft.fft(object_function, simulating_steps) * (1 / simulating_steps)
# Shifting this to the centre at 0
F_object_function = np.fft.fftshift(F_object_function)
# An array of points in the q space, in SI unit
q = 1 / (simulating_steps* (x_array[1] - x_array[0])) * np.arange(0, simulating_steps, 1)
# Shifting the q array to centre at 0 
q = q - np.max(q) / 2


# Taking into account the effect of the contrast aperture    
a = np.sum(np.abs(q) <= q_ap)
if len(q) > a:
    min_index = int(np.ceil(simulating_steps / 2 + 1 - (a - 1) / 2))
    max_index = int(np.floor(simulating_steps / 2 + 1 + (a + 1) / 2))
    q = q[min_index:max_index]
    F_object_function = F_object_function[min_index:max_index]
    
# Arrays for the calculation of the double integration 
Q, QQ = np.meshgrid(q, q)
F_obj_q, F_obj_qq = np.meshgrid(F_object_function, np.conj(F_object_function))

# The modifying function of zeroth-order
R_0 = np.exp(1j*2*np.pi*(C_3*lamda**3 * (Q**4 - QQ**4)/4 + C_5*lamda**5 *(
    Q**6 - QQ**6)/6 - delta_z*lamda*(Q**2 - QQ**2)/2))

# The envelop function by source extension
E_s = np.exp(-np.pi**2/(4*np.log(2)) * q_ill**2 * (C_3*lamda**3 *(
    Q**3 - QQ**3) + C_5*lamda**5 * (Q**5 - QQ**5) - delta_z*lamda*(Q - QQ))**2)

# The envelop function by energy spread
E_cc = (1 - 1j * np.pi/(4*np.log(2)) * C_cc*(delta_E/E)**2 * lamda * (Q**2 - QQ**2))**(-1/2)
E_ct = E_cc * np.exp(-E_cc**2 * np.pi**2/(16*np.log(2)) * (delta_E/E)**2 * (C_c * lamda * (Q**2 - QQ**2) + 1/2*C_3c*lamda**3 * (Q**4 - QQ**4))**2)

AR = np.multiply(np.multiply(np.multiply(np.multiply(F_obj_q, F_obj_qq), R_0), E_s), E_ct)
for i in range(len(q)):
    for j in range(i + 1, len(q)):
        matrixI[:] = matrixI[:] + 2 * (
                AR[j][i] * np.exp(1j * 2 * np.pi * (Q[j][i] - QQ[j][i]) * x_array)).real
    

matrixI[:] = matrixI[:] + np.trace(AR) * np.ones_like(x_array)


matrixI = np.abs(matrixI)

print('Simulation finished.')

t_1 = time.time()

print('Total time:' + str(round((t_1-t_0)/60, 3)) + 'minutes')

##################################
######## End of Main Part ########
##################################




##################################
######## Analysing Results #######
##################################

######## Calculating the resolution for the phase object ########
if object_type == "Step amplitude object" or object_type == "Error function amplitude object":
    half_steps = int(simulating_steps/2)
    I_min = matrixI[half_steps]
    for j in range(half_steps):
        if matrixI[half_steps+j] <= I_min:
            I_min = matrixI[half_steps+j]
            idx_min = half_steps+j
        else:
            break
        
    I_max = matrixI[half_steps]
    for j in range(half_steps):
        if matrixI[half_steps-j] >= I_max:
            I_max = matrixI[half_steps-j]
            idx_max = half_steps-j
        else:
            break
    
    # The region of interest to find the resolution
    x_array_focus = x_array[idx_max:idx_min]
    matrixI_focus = matrixI[idx_max:idx_min]
    
    I_84 = I_min + (I_max - I_min)*84/100
    I_16 = I_min + (I_max - I_min)*16/100
    
    I_84_index = np.where(np.abs(matrixI_focus - I_84) == min(np.abs(matrixI_focus - I_84)))
    x_84 = x_array_focus[I_84_index[0]]
    I_16_index = np.where(np.abs(matrixI_focus - I_16) == min(np.abs(matrixI_focus - I_16)))
    x_16 = x_array_focus[I_16_index[0]]
    resolution = x_16 - x_84
    resolution = resolution[0]

if object_type == "Step phase object" or object_type == "Error function phase object":
    resolution = 0

print("Resolution R = " + str(round(resolution*1e9, 4)) + "nm")

'''
########## Plotting the object ###########
plt.plot(x_array, object_amplitude)
plt.plot(x_array, object_phase)
'''

########## Plotting the curve ############
plt.plot(x_array, matrixI[:, i])

plt.xlim(-10e-9, 10e-9)
# naming the x axis
plt.xlabel('Position x (m)')
# naming the y axis
plt.ylabel('Instensity')
  
# giving a title to my graph
plt.title('I(x)')

plt.show()

################################
###### End of Programme ########
################################
