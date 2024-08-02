#==============================================================================
# This function evaluates the viscosity of the gas mixture via the viscosity of 
# the gas components with their correlation proposed by Perry and Green as flow
#==============================================================================
# from viscosity_i import *
# from matrixC_coefs import *
import numpy as np
import Reactor_catalyst_design as reac

# def viscosity_mixture(Y, T, Wi, Nsp):
#     # matrixC - the matrix of the coefficients for evaluating the viscosity of the species ith
#     # Yi - mole fraction of the the species ith
#     # Mwi - molecular weight of the species ith
#     # Nsp - total number of the species
#     C = np.zeros(4)
#     mu_g = 0
#     matrixC = matrixC_coefs()
#     #print(matrixC)

#     for i in range (0,Nsp):
#         C = matrixC[i][:]
#         mu_i = viscosity_i(C, T)
#         #print(mu_i)
#         sum1 = 0
#         for j in range(0, Nsp):
#             sum1 = sum1 + Y[j]*np.sqrt(Wi[j]/Wi[i])
            
#         mu_g = mu_g + Y[i]*mu_i/sum1
        
#     #return the mixture viscosity 
#     return mu_g

#def viscosity_i(C, T):
#    # C= [C1 C2 C3 C4] are the costants
#    return C[0]*T**(C[1]) / (1.0 + C[2] + C[3]/T**2)

def calc_mug_x(Y,T):
    # matrixC is a Ns x 4 matrix with coefficients for each species
    T_i_x = np.tile(T,(reac.Ns,1)).transpose() # each column is T_x
    T_i_x = T_i_x/1000
    # viscosity of species i (column) at each x-location (row)
    mug_i_x = reac.matrixC_coef[:reac.Ns,0]*np.power(T_i_x,reac.matrixC_coef[:reac.Ns,1]) / ( np.ones_like(T_i_x) + reac.matrixC_coef[:reac.Ns,2] + reac.matrixC_coef[:reac.Ns,3]/T_i_x**2.0 ) # verified against excel
    
    denom = Y @ reac.Mj_by_Mi
    
    # mug_x = Y mug_i_x / denom
    
    mug_x = np.einsum('ij,ij->i',Y,mug_i_x/denom)
    return mug_x
