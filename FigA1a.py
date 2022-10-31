import numpy as np
import matplotlib.pyplot as plt
import time

t0=time.time()

E = 15010  # eV  Nominal Energy After Acceleration
E_0 = 10  # eV  Energy at the sample
alpha_ill = 0.055e-3  # rad Illumination Divergence Angle
lamda = 6.6261e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * E)
lamda_0 = 6.6261e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * E_0)
q_ill = alpha_ill/lamda
sigma_ill = q_ill/(2*np.sqrt(2*np.log(2)))
Scherzer3 = np.sqrt(0.345*lamda/2) 
Scherzer5 = (9/64*92.8*lamda**2)**(1/3)

object_size = 400            # simulating object size in nm
simulating_steps = 1 + 2**11 # total simulating steps
x_array = (np.linspace(-object_size/2, object_size/2, simulating_steps) + object_size/simulating_steps)*1e-9

# A function to choose different LEEM parameters
def choose_LEEM_type(LEEM_type):
    global C_c, C_cc, C_3c, C_3, C_5
    if LEEM_type == 'NAC':
        C_c = -0.075  # m  Second Rank Chromatic Aberration Coefficient
        C_cc = 23.09 # m   Third Rank Chromatic Aberration Coefficient
        C_3c = -59.37  # m   Forth Rank Chromatic Aberration Coefficient
        C_3 = 0.345  # m  Third Order Spherical Aberration Coefficient
        C_5 = 39.4  # m  Fifth Order Spherical Aberration Coefficient
        
    elif LEEM_type == 'AC':
        C_c = 0  # m   Second Rank Chromatic Aberration Coefficient
        C_cc = 27.9 # m   Third Rank Chromatic Aberration Coefficient
        C_3c = -67.4 # m   Forth Rank Chromatic Aberration Coefficient
        C_3 = 0  # m   Spherical Aberration Coefficient
        C_5 = 92.8


q = 1 / (simulating_steps* (x_array[1] - x_array[0])) * np.arange(0, simulating_steps, 1)

# Shifting the q array to centre at 0 
q = q - np.max(q) / 2
# Arrays for the calculation of the double integration 
Q, QQ = np.meshgrid(q, q)

def Ws(delta_z, LEEMType):
    choose_LEEM_type(LEEMType)
    Ws = np.exp(1j*2*np.pi*(C_3*lamda**3 * (Q**4 - QQ**4)/4 + C_5*lamda**5 *(
        Q**6 - QQ**6)/6 - delta_z*lamda*(Q**2 - QQ**2)/2))
    return Ws[int(len(q)/2),:]

def Es(delta_z):
    a_1 = C_3*lamda**3 *(Q**3 - QQ**3) + C_5*lamda**5 * (Q**5 - QQ**5) - delta_z*lamda*(Q - QQ)
    E_s = np.exp(-2*np.pi**2 *sigma_ill**2 *a_1**2)
    return E_s[int(len(q)/2),:]

# A function to calculate the image for single Gaussian distribution
def Ectot(epsilon_n, sigma_E):
    b_1 = 1/2*C_c*lamda*(Q**2 - QQ**2)/E + 1/4*C_3c*lamda**3*(Q**4 - QQ**4)/E
    b_2 = 1/2*C_cc*lamda*(Q**2 - QQ**2)/E**2
    
    d_n = b_1 - 1j*epsilon_n/(2*np.pi*sigma_E**2) 
    
    # The purely chromatic envelop functions
    E_cc = (1 - 1j*4*np.pi*b_2*sigma_E**2)**(-1/2)
    E_ct = E_cc * np.exp(-2*(np.pi*sigma_E*E_cc*d_n)**2 - epsilon_n**2/(2*sigma_E**2))
    
    E_ctot = E_ct[int(len(q)/2),:]
    return E_ctot

def EcFN():
    return 0.28682*Ectot(-0.03925, 0.0531)+ 0.33146*Ectot(-0.3382, 0.1991) + 0.38173*Ectot(-0.1438, 0.0962)

def R_G1(delta_z, LEEMType):
    choose_LEEM_type(LEEMType)
    delta_E = 0.2424 
    sigma_E = delta_E/(2*np.sqrt(2*np.log(2)))
    return Ws(delta_z, LEEMType)*Es(delta_z)*Ectot(0,sigma_E)

def R_FE(delta_z, LEEMType):
    choose_LEEM_type(LEEMType)
    return Ws(delta_z, LEEMType)*Es(delta_z)*EcFN()


q=q/1e9
q_apNAC = 2.34e-3/lamda 
q_apAC = 7.37e-3/lamda 
# xticks = [0, 0.5, 1, 1.5]
yticks = 0.2*np.arange(-1,7)
alpha_ap_list = [2.51e-3, 2.34e-3]
# defocus_list = [0.75e-6,0]

CTF = R_FE(0, 'NAC')
CTFG = R_G1(0, 'NAC')
W_s = Ws(0, 'NAC')

fig, ax = plt.subplots(2,1)
ax1,ax2 = ax
fig.set_size_inches(4.8, 5.2)
fig.tight_layout(rect=[0.09, 0.05, 1, 1])
fig.subplots_adjust(hspace=0)

s=15
width=2

ax1.plot(q, W_s.real, 'k-', linewidth=width, label = r'Re[$\mathrm{W_s(q,0)}$]')
ax1.plot(q, CTFG.real, 'b-', linewidth=width, label = r'Re[$\mathrm{R(q,0,0)}$]')
ax1.plot(q, CTF.real, 'r-', linewidth=width, label = r'Re[$\mathrm{R_{(N)}(q,0,0)}$]')

ax2.plot(q, W_s.imag, color = 'gray', linewidth=width, label = r'Im[$\mathrm{W_s(q,0)}$]')
ax2.plot(q, CTFG.imag, 'g-', linewidth=width, label = r'Im[$\mathrm{R(q,0,0)}$]')
ax2.plot(q, CTF.imag, color = 'darkorange', linewidth=width, label = r'Im[$\mathrm{R_{(N)}(q,0,0)}$]')


ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.9)

ax2.set_xlabel(r'q $\mathrm{(nm^{-1})}$', fontsize=s)

yticks= np.array([-1. , -0.5,  0. ,  0.5,  1. ])

for i in range(2):
    ax[i].minorticks_on()
    ax[i].set_yticks(yticks)
    ax[i].set_yticklabels(yticks, fontsize=s-2)
    ax[i].axvline(x=q_apNAC/1e9, color='k', linestyle=':')
    ax[i].set_ylabel('Amplitude', fontsize=s)
    ax[i].axhline(y=0, color='k', linestyle='-', linewidth=0.9)
    ax[i].set_ylim(-1.3, 1.2)
    ax[i].set_xlim(0, 0.45)

xticks=[0, 0.1, 0.2, 0.3, 0.4]
ax1.set_xticks([])
# ax.set_xticklabels(xticks, fontsize=s-2)

ax2.set_xticks(xticks)
ax2.set_xticklabels(xticks, fontsize=s-2)

ax1.text(-.096, 1.1, '(a)',fontsize=s)

# ax.legend(frameon=False, fontsize=s-3)

# fig, ax = plt.subplots(nrows=1, ncols=2)
# fig.set_size_inches(10, 4)
# fig.tight_layout(rect=[0.06, 0.03, 1.02, 0.95])
# fig.subplots_adjust(wspace=0.4)

# ax1,ax2 =ax
# s=15

# for i in range(2):
#     ax[i].axhline(y=0, color='k', linestyle='-', linewidth=0.9)
#     ax[i].set_ylim(-0.2, 1.2)
#     ax[i].set_xlabel(r'q $\mathrm{(nm^{-1})}$', fontsize=s)
#     ax[i].minorticks_on()
#     ax[i].set_yticks(yticks)
#     ax[i].set_ylabel('Amplitude', fontsize=s)
#     CTF = R_G1(alpha_ap_list[i], defocus_list[i], 'NAC')
#     ax[i].plot(q/(1e9), CTF.real, 'r-', linewidth=2, label = r'Re[R]')
#     ax[i].plot(q/(1e9), CTF.imag, 'b-', linewidth=2, label = r'Im[R]')


# ax1.set_xlim(0, 0.6)
# ax2.set_xlim(0, 1.8)
# ax1.axvline(x=q_apNAC/1e9, color='k', linestyle=':')
# ax2.axvline(x=q_apAC/1e9, color='k', linestyle=':')
# ax1.text(0.52, 1.08, s = 'NAC', color = 'k', fontsize=s)
# ax2.text(1.62, 1.08, s = 'AC', color = 'k', fontsize=s)
# ax1.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
# ax2.set_xticks([0, 0.5, 1, 1.5])
# ax1.text(0.01,0.35, r"$\mathrm{Im(E_C)}$", color='k' ,fontsize=s)
# ax2.text(0.25,0.25, r"$\mathrm{Im(E_C)}$", color='k' ,fontsize=s)
# ax1.text(0.03,1.05, r"$\mathrm{Re(E_C)}$", color='k' ,fontsize=s)
# ax2.text(0.15,1.05, r"$\mathrm{Re(E_C)}$", color='k' ,fontsize=s)

# ax1.text(-0.15,1.15, '(a)',fontsize=s)
# ax2.text(-0.45,1.15, '(b)',fontsize=s)


# handles, labels = ax2.get_legend_handles_labels()
# # fig.legend(handles, labels, loc=8, ncol = 4, frameon=False, fontsize=s)
# fig.subplots_adjust(bottom=0.2)

print(time.time() - t0)

plt.savefig("Figure_A1a.png", dpi=1000)