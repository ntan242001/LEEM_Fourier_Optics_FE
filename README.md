
# Fourier Optics for Image Formation in LEEM

## Description
This project contains programmes that simulate and analyse images formed by Low Energy Electron Microscopy (LEEM). The principle and calculations are based on the following articles: 
* [K.M.Yu et al 2019 Ultramicroscopy 200, 160-168](https://doi.org/10.1016/j.ultramic.2019.01.015)
* [A B Pang et al 2009 J. Phys.: Condens. Matter 21 314006](https://doi.org/10.1088/0953-8984/21/31/314006)
* [S.M.Schramm et al 2012 Ultramicroscopy 115, 88-108](https://doi.org/10.1016/j.ultramic.2011.11.005)


## Installation
The programmes are written in Python 3 with common libraries such as numpy, matplotlib. Users are recommended to install Anaconda and edit the programmes with Jupyter Notebook or Spyder.

## LEEM Fourier Optics of One Dimensional Objects
### Intensity profile
The programme "Main_1D_1Data.py" provides a plot of intensity versus position, as well as the microscope's resolution. Users can choose between different LEEM constants, defocus modes and object functions. To learn the programme in more details, users can run the file "Main_1D_1Data.ipynb" in Jupyter Notebook.

### Resolution as a function of aperture angle
The programme "Aperture_Optimal_Resolution.py" helps calculate the microscopy's resolution as a function of the contrast aperture angle. After running this programme (which takes a relatively long time (an hour for 30 data in range (1 mrad, 10 mrad)) for Dell Core i5 Pro Desktop), users will attain a csv file containing the calculated array R(alpha_ap).

### Resolution as a function of initial energy
The programme "R(E_0).py" helps calculate the microscopy's resolution as a function of initial energy E_0. After running this programme (which takes a relatively long time (an hour for 30 data in range (10 eV, 100 eV)) for Dell Core i5 Pro Desktop), users will attain a csv file containing the calculated array R(E_0).

### Intensity profile with second order Taylor expansion
The programme "2nd_Order_Taylor.py" takes into account the second order expansion of the small terms k/q (in source extension) and deviation from the nominal energy (in electron energy distribution).

## Contributing
For contribution request, please email the author at xtnguyenaa@connect.ust.hk.

## Authors and acknowledgement
This is part of the UROP project on LEEM Fourier Optics at HKUST in summer 2021 of the author under the supervision of professor M.S. Altman. 

## Project status
Active
