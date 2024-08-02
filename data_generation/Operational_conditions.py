#-------------------------------------------------------------------------------
# OPERATIONAL CONDITIONS 
#-------------------------------------------------------------------------------
import numpy as np
import Reactor_catalyst_design as reac
from Molecular_weight_i import*

Wi = Molecular_weight_i()

R      = 8.3145                             #Universal gas constant
P_std  = 101325.0                           #Pressure of the gas at standard condition
T_std  = 273.15                             #Temperature of the gas at standard condition  

T0  = 300.0                                 #Temperature at reactor inlet, K
Tf  = 650+273.15                                #Temperature of the heat jacket, K 
P0  = 101325.0                              #Pressure at the reactor inlet, Pa

#Define the feeding at standard conditions 
# alternative definition of input flow
Q_pump1 = 1.0                           #Ammonia dioxide flow rate kg/day
Q_pump2 = Q_pump1*0.7/0.3               #Argon pump flow rate kg/day

f_NH3   = Q_pump1/Wi[0]/24/60/60        #gas molar of CO2, mol/s
f_AR    = 0.0#f_NH3*0.7/0.3                 #gas molar of H2, mol/s
f_t     = f_NH3 + f_AR                  #total gas molar, mol/s

#Volume of the f_t mole of the gas at standard condition 
Volume = (f_NH3*Wi[0] + f_AR*Wi[3])*R*T_std/P_std

#Space velocity at the standard condition (273.15K, 101325 Pa)
SV0 = Volume/reac.Ac                                            
#print('Space velocity at standard condition:',SV0)

#1. Gas velocity along the reactor, ug (m/s)
from Vel_gasmixture import*
ug = Vel_gasmixture(T0, P0, SV0, reac.Din, reac.Dp, reac.mc)
# print('Space velocity at working condition:',ug)

#2. Resident time of the gas flow inside reactor (s)
t_final = reac.L/ug      

#3. Axial mass dispersion term
Da = 0.001 #Noted for further implementation 

#4. Axial thermal dispersion term
lamdaa = 0.001 #Noted for further implementation 
