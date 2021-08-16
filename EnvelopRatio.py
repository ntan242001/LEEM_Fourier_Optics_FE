# Comparing the single and double Gaussian distributions for different aperture angles
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time
import csv
import math

##############################
######### Preamble ###########
##############################


# A function to choose different LEEM parameters
def choose_LEEM_type(LEEM_type_str, aberration_corrected_bool = False):
    global E, E_0, C_c, C_cc, C_3c, C_3, C_5, alpha_ap, alpha_ill, \
        delta_E, M_L, lamda, lamda_0, q_ill, q_ap, LEEM_type, aberration_corrected
    LEEM_type = LEEM_type_str
    aberration_corrected = aberration_corrected_bool
    
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
            print("IBM nac chosen.")
            
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
            print("IBM ac chosen.")
            
        lamda = 6.6261e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * E)
        lamda_0 = 6.6261e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * E_0)
    
        q_ap = alpha_ap/lamda
        #q_ill = alpha_ill/lamda
        
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
            
            lamda = 6.6261e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * E) # in metre
            alpha_ap = (lamda/C_3)**(1/4) # rad Aperture angle for optimal resolution
            
            print("Custom nac LEEM at E_0 = " + str(E_0) + " eV chosen.")
            
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
            
            lamda = 6.6261e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * E) # in metre
            alpha_ap = (3/2*lamda/C_5)**(1/6) # rad Aperture angle for optimal resolution
            
            print("Custom ac LEEM at E_0 = " + str(E_0) + " eV chosen.")   
        
        lamda_0 = 6.6261e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * E_0) # in metre
        
        q_ap = alpha_ap/lamda
        #q_ill = alpha_ill/lamda


# A function to set different defocus values
def choose_defocus(defocus_type, value = 0):
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
    elif defocus_type == "custom":
        delta_z = value


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
    print(object_type + " created")

choose_LEEM_type("IBM", aberration_corrected_bool = True)
choose_defocus("In-focus")
create_object("Step phase object", k = 1)

##################################
######## End of Preamble #########
##################################



##################################
########### Main Part ############
##################################
print("Simulation start.")
t_0 = time.time()
# The object image is reversed through the lens
object_function_reversed = object_function[::-1] 
    
# Creating an array of different cut-off frequencies
alpha_ill = 10*1e-3

q_ill= alpha_ill/lamda


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
    
# Arrays for the calculation of the double integration 
Q, QQ = np.meshgrid(q, q)

# A function to calculate the image for single Gaussian distribution
#def Image1(q_ill, q_ill_index):
sigma_E = delta_E/(2*np.sqrt(2*np.log(2)))
sigma_ill = q_ill/(2*np.sqrt(2*np.log(2)))

a_1 = C_3*lamda**3 *(Q**3 - QQ**3) + C_5*lamda**5 * (Q**5 - QQ**5) - delta_z*lamda*(Q - QQ)

b_1 = 1/2*C_c*lamda*(Q**2 - QQ**2)/E + 1/4*C_3c*lamda**3*(Q**4 - QQ**4)/E
b_2 = 1/2*C_cc*lamda*(Q**2 - QQ**2)/E**2

# The envelop function by source extension
E_s = np.exp(-2*np.pi**2 *sigma_ill**2 *a_1**2)

# The purely chromatic envelop functions
E_cc = (1 - 1j*4*np.pi*b_2*sigma_E**2)**(-1/2)
E_ct = E_cc * np.exp(-2*np.pi**2 *E_cc**2 *sigma_E**2 *b_1**2)

E1 = np.multiply(E_s, E_ct)


#def Image2(q_ill, q_ill_index):    
a_1 = C_3*lamda**3 *(Q**3 - QQ**3) + C_5*lamda**5 * (Q**5 - QQ**5) - delta_z*lamda*(Q - QQ)
a_2 = 3/2*C_3*lamda**3 *(Q**2 - QQ**2) + 5/2*C_5*lamda**5 * (Q**4 - QQ**4) 

sigma_E = delta_E/(2*np.sqrt(2*np.log(2)))
sigma_ill = q_ill/(2*np.sqrt(2*np.log(2)))

b_1 = 1/2*C_c*lamda*(Q**2 - QQ**2)/E + 1/4*C_3c*lamda**3*(Q**4 - QQ**4)/E
b_2 = 1/2*C_cc*lamda*(Q**2 - QQ**2)/E**2
b_3 = C_c*lamda*(Q - QQ)/E + C_3c*lamda**3*(Q**3 - QQ**3)/E

# The purely chromatic envelop functions
E_cc = (1 - 1j*4*np.pi*b_2*sigma_E**2)**(-1/2)
E_ct = E_cc * np.exp(-2*np.pi**2 *E_cc**2 *sigma_E**2 *b_1**2)

d_1 = a_1 + 1j*2*b_1*b_3*sigma_E**2*E_cc**2
d_2 = a_2 + 1j*b_3**2*sigma_E**2*E_cc**2

# The mixed envelop function 
E_m = (1 - 1j*4*np.pi*d_2*sigma_ill**2)**(-1/2)
E_mt = E_m * np.exp(-2*np.pi**2 *E_m**2 *sigma_ill**2 *d_1**2)

E2 = np.multiply(E_mt, E_ct)

gamma = np.abs(E2/E1)

print('Simulation finished.')
t_1 = time.time()

print('Total time: ' + str(round((t_1-t_0)/60, 3)) + ' minutes')

##################################
######## End of Main Part ########
##################################



##################################
######## Analysing Results #######
##################################
'''
# Plotting the object
plt.plot(x_array, object_amplitude)
plt.plot(x_array, object_phase)
'''
q_centre = 292
delq = 170

q_min = q_centre - delq
q_max = q_centre + delq

# plotting the curve
plt.pcolor(q[q_min:q_max]/1e9, q[q_min:q_max]/1e9, gamma[q_min:q_max, q_min:q_max], shading='auto', cmap = 'jet')

plt.xticks(np.arange(-0.4, 0.5, step=0.2))
plt.yticks(np.arange(-0.4, 0.5, step=0.2))

plt.xlabel("$q$ ($nm^{-1}$)")
plt.ylabel("$q'$ ($nm^{-1}$)")

#plt.xlim(-23* 10**6, 23* 10**6)
#plt.ylim(-23* 10**6, 23* 10**6)
  
# giving a title to my graph
plt.title('$\Delta z = 0$, $\\alpha_{ill}$ = ' + str(round(alpha_ill*1e3, 3)) + ' mrad')
plt.colorbar()

plt.show()
