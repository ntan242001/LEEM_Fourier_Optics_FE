
# Fourier Optics for Image Formation in LEEM

## Description
This project contains programmes that simulate and analyse images formed by Low Energy Electron Microscopy (LEEM). The principle and calculations are based on the following articles: 
* [K.M.Yu et al 2019 Ultramicroscopy 200, 2019, 160-168](https://doi.org/10.1016/j.ultramic.2019.01.015)
* [A B Pang et al 2009 J. Phys.: Condens. Matter 21 314006](https://doi.org/10.1088/0953-8984/21/31/314006)


## Installation
The programmes are written in Python 3 with common libraries such as numpy, matplotlib. Users are recommended to install Anaconda and edit the programmes with Jupyter Notebook or Spyder.

## LEEM Fourier Optics of One Dimensional Objects
### Intensity profile
The programme "Main_1D_1Data.py" provides a plot of intensity versus position, as well as the microscope's resolution. Users can choose between different LEEM constants, defocus modes and object functions. To learn the programme in more details, users can run the file "Main_1D_1Data.ipynb" in Jupyter Notebook.

### Resolution as a function of aperture angle


### Resolution as a function of initial energy


### Intensity profile with second order Taylor expansion
The programme "2nd_Order_Taylor.py" takes into account the second order expansion of the small terms $\frac{k}{q}$ and $\frac{\varepsilon}{E}$

## Contributing

## Authors and acknowledgement

## Project status
