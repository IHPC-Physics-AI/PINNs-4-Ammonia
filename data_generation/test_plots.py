import numpy as np
import matplotlib 
from scipy.integrate import odeint
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import Reactor_catalyst_design as reac
import Operational_conditions as oper


V = np.load('Solution.npy')

YiT2000 = np.reshape(V[19999][:], (reac.Nx, reac.Ns+1), order='F')
x = np.linspace(0.0,reac.L,reac.Nx)

for i in range(0,reac.Ns+1):    
    plt.figure()
    if(i<reac.Ns):
        plt.plot(x,100*YiT2000[:,i],'r-')
    else:
        plt.plot(x,YiT2000[:,i],'r-')
    plt.xlabel('Along X-direction')
    if i==0:
        plt.ylabel('Mole fraction of CO2')
    if i==1:
        plt.ylabel('Mole fraction of H2')
    if i==2:
        plt.ylabel('Mole fraction of CO') 
    if i==3:
        plt.ylabel('Mole fraction of H2O')
    if i==4:
        plt.ylabel('Mole fraction of CH4')
    if i==5:
        plt.ylabel('Mole fraction of C2H6')
    if i==6:
        plt.ylabel('Mole fraction of C3H8') 
    if i==7:
        plt.ylabel('Mole fraction of C4H10')
    if i==8:
        plt.ylabel('Mole fraction of CH2')
    if i==9:
        plt.ylabel('Mole fraction of C2H4')
    if i==10:
        plt.ylabel('Mole fraction of C3H6') 
    if i==11:
        plt.ylabel('Mole fraction of C4H8')
    if i==12:
        plt.ylabel('Temperature, (K)')
    plt.grid(True)
plt.show()
