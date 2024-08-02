#===============================================================
# Function defines the information of the reactor and catalyst
# pellet
#===============================================================

import numpy as np
from Molecular_weight_i import*

L  = 1.0       # Reactor length, m
D0 = 2.54e-2    # Reactor outer diameter, m
Din = 2.0e-2    # Reactor inner diameter, m
Dp = 0.0015     # Catalyst pellet diameter, m, originally 0.0015
Df = D0+0.002*2.0   # Furnace diameter / heat jacket diameter, m
rho_b = 904.0   # Packing density, kg/m^3, originally 904
eta = 0.99      # Catalyst effectiveness factor

Nx = 51        # Number of discrete point along the reactor
dx = L/(Nx-1)   # size of an element, m
xvals = np.linspace(0,L,Nx) # Number of the discrete locations 


#1. Compute the bed porosity
#from Bed_porosity import*
epsilon_b = 0.38 + 0.073*(1.0 - (D0/Dp - 2.0)**2/(D0/Dp)**2)
#Bed_porosity(Din, Dp) #0< epsilon < 1
#print('epsilon = ', epsilon_b)

#2. Reactor crossed section area
Ac = np.pi*(Din/2)**2
#print('Ac = ', Ac)

#3. Density of the catalyst kg/m^3
rho_p = rho_b/(1.0 - epsilon_b)
#print('rho_p = ',rho_p)

#4. Total catalyst weight inside the reactor
mc = rho_b*Ac*L


############################################################################
# 4-species chemical reaction scheme 
    
# kinetic constants
Ns = 4          #Total number of species
N  = Nx*(Ns+1)  #Total variable of the large matrix 
Nr = 1          #Number of reaction 

#1. Stoichiometric coefficient
#[NH3,  H2,  N2, Ar]  
#[  |   |    |    |]     
#[  0,  1,   2,   3]    

## Stoichiometric coefficient
stoi_vij = np.zeros(Ns)
stoi_vij[0] = -2.0
stoi_vij[1] = 3.0
stoi_vij[2] = 1.0

# Heat released from chemical reaction
DHr = -45600.0

# species properties: Polar or nonpolar
fs  = [0.733,
       1.0,
       1.0,
       1.0]

### mixture viscosity subroutine
matrixC_coef =   np.array([[7.24238380e-03, 9.40613857e-01, 2.01052920e+02, 2.69408538e+00],         #NH3  [0]
                 [2.65255557e-02,  -1.30542430e+00,  -9.37838615e+00, 1.28838353e+03],               #H2   [1]
                 [6.89400597e-03, 6.44713803e-01, 1.63910616e+02, 1.09061213e+00],                   #N2   [2]
                 [8.46258883e-04, 6.45185645e-01, 1.40300927e+01, 1.91297973e-01]])                  #AR   [3]


Mj_by_Mi = np.zeros((Ns,Ns))
for ii in range(0,Ns):
    for jj in range(0,Ns):
        Mj_by_Mi[ii,jj] = (Molecular_weight_i()[ii] / Molecular_weight_i()[jj])**0.5
        # the (i,j) element is Mi divided by Mj

#Builing Temperature
Si = 1.5*np.ones(Ns)
Tb = [239.8, 20.3, 77.34, 87.2]
fs = [0.733, 1.0, 1.0, 1.0]
Si = Si*Tb
Si[1] = 79

#print(Si)

Si_j = np.zeros((Ns,Ns))
for i in range(0,Ns):
    for j in range(0,Ns):
        Si_j[i,j] = fs[i]*np.sqrt(Si[i]*Si[j])
#print(Si_j)

### heat capacity routine: Corrected
matrixA_coef = np.array([[19.99563, 49.77119, -15.37599, 1.921168, 0.189174],                       #NH3
                 [33.066178,  -11.363417, 11.432816,  -2.772874,  -0.158558],                       #H2             
                 [19.505830,  19.887050,  1.3697840,   0.527601,  -4.935202],                       #N2                   
                 [20.78600, 2.825911E-7, -1.464191E-7, 1.092131E-8, -3.661371E-8]])                 #AR                 

## 
matrixB_coef  = np.array([[0.70717629, 0.87712917, 4.09458442, -0.4458513],                         #NH3  [0]
                     [3.42961237e+00,  -2.90547479e-07, 7.33928075e+00,  1.93431432e-01],           #H2   [1]
                     [1.79616324e+00, -2.72877800e-07, 2.70845470e+01, 3.99694386e-01],             #N2   [2]
                     [1.37446554e+00, -1.20619828e-06, 3.44051496e+00, 2.81153756e+01]])            #AR   [3]


# Diffusion matrix using fourth-order central differencing
d2_dx2 = np.zeros((Nx,Nx))
for i in [1,Nx-2]:
    d2_dx2[i][i]   = -2.0 / dx**2
    d2_dx2[i][i-1] = 1.0 / dx**2
    d2_dx2[i][i+1] = 1.0 / dx**2

denom4CFD = 12*dx**2 # denominator for 4th-order CFD
for i in range(2,Nx-2):
    d2_dx2[i][i-2] = -1 / denom4CFD
    d2_dx2[i][i-1] = 16 / denom4CFD
    d2_dx2[i][i] = -30 / denom4CFD
    d2_dx2[i][i+1] = 16 / denom4CFD
    d2_dx2[i][i+2] = -1 / denom4CFD
    
#  Advection matrix using 1st order upwind (only positive flow)
d_dx = np.zeros((Nx,Nx))  
for i in range(1,Nx-1):
    d_dx[i][i] = 1. / dx
    d_dx[i][i-1] = -1. / dx
