# Energy dependence of 1D pure amplitude object with varying initial electron 
## energy in nac LEEM at in-focus condition

import numpy as np
import matplotlib.pyplot as plt
import time

t_0 = time.time()

object_size = 400               # simulating object size in nm
simulating_steps = 1 + 2**10    # total simulating steps

# A function to choose different LEEM parameters
def choose_LEEM_type(LEEM_type_str):
    global E, E_0, C_c, C_cc, C_3c, C_3, C_5, alpha_ap, alpha_ill, \
        delta_E, M_L, lamda, lamda_0, q_ill, q_ap, LEEM_type
    LEEM_type = LEEM_type_str   
    
    if LEEM_type != "Energy dependent":
        if LEEM_type == "IBM AC":
            E = 15010  # eV  Nominal Energy After Acceleration
            E_0 = 10  # eV  Starting Energy
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
    
        elif LEEM_type =="IBM NAC":
            E = 15010  # eV  Nominal Energy After Acceleration
            E_0 = 10  # eV  Starting Energy
            C_c = -0.075  # m   Chromatic Aberration Coefficient
            C_cc = 23.09 # m   Third Rank Chromatic Aberration Coefficient
            C_3c = -59.37  # m   Forth Rank Chromatic Aberration Coefficient
            C_3 = 0.345  # m  Third Order Spherical Aberration Coefficient
            C_5 = 39.4  # m  Fifth Order Spherical Aberration Coefficient
            
            alpha_ap = 2.34e-3  # rad Aperture angle
            alpha_ill = 0.1e-3  # rad Illumination Divergence Angle
            
            delta_E = 0.25  # eV  Energy Spread
            M_L = 0.653  # Lateral Magnification
            print("IBM NAC chosen.")
        
        lamda = 6.6261e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * E)
        lamda_0 = 6.6261e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * E_0)
    
        q_ap = alpha_ap / lamda
        q_ill = alpha_ill / lamda
        
    elif LEEM_type == "Energy dependent":
        E = 15010  # eV  Nominal Energy After Acceleration
        alpha_ill = 0.1e-3  # rad Illumination divergence angle
        
        delta_E = 0.25  # eV  Energy Spread
        M_L = 0.653  # Lateral Magnification
        
        lamda = 6.6261e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * E) #in metre
        q_ill = alpha_ill / lamda
        print("Energy dependent simulation chosen.")

choose_LEEM_type("IBM NAC")

# Creating a 1:1/sqrt(2) amplitude object whose phase is uniformly set to 0
# An array of points in the x space
x_array = np.linspace(-object_size/2, object_size/2, simulating_steps)

object_phase = np.zeros_like(x_array)

object_amplitude = np.ones_like(x_array)
for counter, element in enumerate(x_array):
    if element > 0:
        object_amplitude[counter] = 1/np.sqrt(2)
        
# Object function
object_function = np.multiply(object_amplitude, np.exp(1j * object_phase))
# The object image is reversed through the lens
object_function = object_function[::-1]         

# The Fourier Transform of the Object Wave Function
F_object_function = np.fft.fft(object_function, simulating_steps) * (1 / simulating_steps)
F_object_function = np.fft.fftshift(F_object_function)
# The complex conjugate of the Fourier Transform of the Object Wave Function
F_object_function_conj = np.conj(F_object_function)
# An array of points in the q space, in SI unit
q_array = 1 / (object_size * 1e-9) * np.arange(0, simulating_steps, 1)
q_array = q_array - np.max(q_array) / 2     # shift the q array to centre at 0 


"""# Creating an E_0 array from 5eV to 100eV 
E_0_array = np.linspace(5, 100, 2**10)

for index in range(len(E_0_array)):    
    E_0 = E_0_array[index]
    kappa = np.sqrt(E/E_0)
    # Third Order Spherical Aberration Coefficient in metre
    C_3 = 0.0327 * np.sqrt(kappa) + 0.1681  
    # Chromatic Aberration Coefficient in metre
    C_c = -0.0154 * np.sqrt(kappa) + 0.0173 
    # Maximum spacial frequency in metre^(-1)
    q_ap = 1/(C_3*lamda**3)**(1/4)
""" 

# Initialising the intensity array I(x)
I = np.zeros_like(x_array, dtype=complex)

# Calculating the array of aperture function M(q)
for i in range(len(q_array)):
    q = q_array[i]
    M = np.ones_like(q_array)
    if np.abs(q) > q_ap:
        M[i] = 0

for k in range(len(I)):
    for i in range(len(q_array)):
        # Calculating the wave aberration function W_s(q)
        W_s_q = np.exp(1j*2*np.pi*1/4*C_3*(lamda**3)*(q**4))   
    
        for j in range(len(q_array)):
            qq = q_array[j]
            
            # Calculating the wave aberration function W_s*(q')
            W_s_qq = np.exp(-1j*2*np.pi*1/4*C_3*(lamda**3)*(qq**4))  
            # The function R_0(q, q')
            R_0 = M[i]*M[j]*W_s_q*W_s_qq
            
            # The envelop function by source extension
            E_s = np.exp(-(np.pi**2)/(4 * np.log(2)) * q_ill**2 * (
            C_3 * lamda**3 * (q**3 - qq**3))** 2 )
            
            # The envelop function by energy spread
            E_c = np.exp(-(np.pi**2)/(16*np.log(2)) * (delta_E/E)**2 * 
                         (C_c * lamda * (q**2 - qq**2))**2)
            
            # The function R
            R = R_0*E_s*E_c
            
            I[k] += F_object_function[i] *F_object_function_conj[j]*R*np.exp(1j*2*np.pi*(q-qq)) * (q_array[1] - q_array[0])**2
    
t_f = time.time()

print(t_f-t_0)    
    
# plotting the points 
plt.plot(x_array, I)
  
# naming the x axis
plt.xlabel('Position x (nm)')
# naming the y axis
plt.ylabel('Instensity $\frac{W}{m^2}$')
  
# giving a title to my graph
plt.title('I(x)')
  
# function to show the plot
plt.show()

