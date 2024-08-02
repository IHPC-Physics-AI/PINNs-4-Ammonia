#==============================================================================
# This function evaluates the heat capacity of the gas mixture via the estimated
# correlation 
#==============================================================================
import numpy as np
import Reactor_catalyst_design as reac

        
def calc_Cpg_x(Y,T):
    # T is temperature at all x-locations
    # Y is composition (columns) at each x-location (row)
    
    T_i_x = np.tile(T,(reac.Ns,1)).transpose() # each column is T_x
    t = T_i_x/1000
    
    Cpg_i_x = reac.matrixA_coef[:reac.Ns,0] + reac.matrixA_coef[:reac.Ns,1]*t + reac.matrixA_coef[:reac.Ns,2]*t**2 + reac.matrixA_coef[:reac.Ns,3]*t**3 + reac.matrixA_coef[:reac.Ns,4]/t**2
    Cpg_x = np.einsum('ij,ij->i',Y,Cpg_i_x) # rowwise dot product of composition and molar heat capacity
    if np.any(Cpg_x < 0):
        print("negative Cpg_x")
        print("T: ", T)
        print("Y: ", Y)
    return Cpg_x
