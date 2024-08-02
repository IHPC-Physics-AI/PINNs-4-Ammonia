#===========================================================================
# This function defines the reaction rate of the chamical raction 
#===========================================================================
import numpy as np
import Reactor_catalyst_design as reac
import Operational_conditions as oper
import math


def calc_reaction_x(T, P, UP, Tf, rho_b, eta):

    #Ns     - number of species in the chemical process
    #Nr     - number of rection in the chemical process
    #niu_bf - stoichiometric coefficients
    #DHr    - heat realsed from reaction r
    #T      - Temperature
    #R      - total pressure at point
    #R      - gas constant 
    #======================================================================

    # read info from class
    Ns = reac.Ns
    Nr = reac.Nr

    #Assign parameters
    #Pt  = 101.325 
    A1  = 36180.0
    A2  = 8730.0
    Ea1 = 82600
    Ea2 = 76710
    
    a1  = 1.5
    a2  = 0.5
    a3  = 1.0
    
    a41 = 0.24
    a42 = 0.28
    a51 = -0.54
    a52 = -0.42

    
    stoi_vij = reac.stoi_vij
    DHr = reac.DHr
    R = oper.R

    #Partial pressure in Pa
    P_NH3 = P[:,0]/1000.0
    P_H2  = P[:,1]/1000.0
    P_N2 = P[:,2]/1000.0
    P_AR = P[:,3]/1000.0

    Kr_x = np.zeros(reac.Nx)
    Ri_x = np.zeros(reac.Nx)
    r_Yi_x = np.zeros((reac.Nx,Ns))    
    SDHr  = np.zeros(reac.Nx)
    
    #Reaction rate
    for i in range (0, reac.Nx):
        if (P_NH3[i]<1.0e-6):
            Ri_x[i] = 0
        else:

            #Calculate the equilibrium coefficient 
            log10Kp_star = -2.691122*np.log10(T[i]) - 5.519265E-5*T[i] + 1.848863E-7*T[i]*T[i] + 2001.6/T[i] + 2.6899
            log10Kp = log10Kp_star + (0.1191849/T[i] + 91.87212/np.power(T[i],2) + 25122730.0/np.power(T[i],4))#*oper.P0/101325.0
            Kp = np.power(10.0,log10Kp)
        
            beta = Kp*np.power(P_H2[i],a1)*np.power(P_N2[i],a2)/np.power(P_NH3[i],a3) # pressure in kPa - might need to check

            if (T[i] < 748):
                Ri_x[i] = A1*np.exp(-Ea1/T[i]/R)*np.power(P_NH3[i],a41)*np.power(P_H2[i],a51)*(1.0 - np.power(beta,2)) # pressure in kPa
            else: 
                Ri_x[i] = A2*np.exp(-Ea2/T[i]/R)*(P_NH3[i]**a42)*(P_H2[i]**a52)*(1.0 - beta**2) # pressure in kPa
        
    # Energy balance
    SDHr  = SDHr + Ri_x*DHr
    
    # for each reaction, add the species formation
    r_Yi_x += np.outer(Ri_x,stoi_vij[:])
    
    #print(np.reshape(r_Yi_x, (reac.Ns*reac.Nx,1),order='F'))
    # print(">> SDHr: ", SDHr)
    # print("[Kinetic_model] >> dCi: ", dCi)
    
    dCi = np.zeros(reac.N)
    dCi[:reac.Ns*reac.Nx] = np.reshape(r_Yi_x, (reac.Ns*reac.Nx),order='F')
    
    dCi[reac.Ns*reac.Nx:] = -rho_b * eta * SDHr + UP*(Tf-T) #in the case of adiabatic please comment out the term "UP*(Tf-T)"
    return dCi









    
