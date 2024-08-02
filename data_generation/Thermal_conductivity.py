#==============================================================================
# This function evaluates the thermal conductivity of the gas mixture via the
# estimated correlation based on Lindsay and Bromley 
#==============================================================================
import numpy as np
from Molecular_weight_i import*
import Reactor_catalyst_design as reac



#     # Return the mixture thermal conductivity
#     return Kw

# NH3 - polar molecules   , fs = 0.733
# H2  - nonpolar molecules, fs = 1
# N2  - nonpolar molecules, fs = 1
# Ar  - nonpolar molecules, fs = 1

## latest optimal version
def calc_kg_x_WA(Y, T):
    T_i_x = np.tile(T, (reac.Ns, 1)).transpose() / 1000  

    # Viscosity of the components
    viscosity_i_x = reac.matrixC_coef[:reac.Ns, 0] * np.power(T_i_x, reac.matrixC_coef[:reac.Ns, 1]) / (
        np.ones_like(T_i_x) + reac.matrixC_coef[:reac.Ns, 2] + reac.matrixC_coef[:reac.Ns, 3] / T_i_x ** 2.0)

    # Thermal conductivity of the components
    kg_i_x = reac.matrixB_coef[:reac.Ns, 0] * np.power(T_i_x, reac.matrixB_coef[:reac.Ns, 1]) / (
        np.ones_like(T_i_x) + reac.matrixB_coef[:reac.Ns, 2] / T_i_x + reac.matrixB_coef[:reac.Ns, 3] / T_i_x ** 2.0)

    # Evaluate the Sutherland coefficient of species i-th
    sum1 = np.zeros((reac.Nx, reac.Ns))

    for k in range(reac.Nx):
        YY = Y[k][:reac.Ns]
        viscosity_ratio = np.divide.outer(viscosity_i_x[k], viscosity_i_x[k])
        sqrt_term = np.sqrt((T_i_x[k] + reac.Si) / (T_i_x[k] + reac.Si.T))
        value4 = (1.0 + np.sqrt(viscosity_ratio * (reac.Mj_by_Mi ** 0.75) * sqrt_term)) ** 2
        value5 = (T_i_x[k][:, np.newaxis] + reac.Si_j) / (T_i_x[k][:, np.newaxis] + reac.Si)
        Theta_ij = 0.25 * value4 * value5
        sum1[k] = np.einsum('i,ij->j', YY, Theta_ij)

    # Evaluate the thermal conductivity of the mixture
    kg_x = np.einsum('ij,ij->i', Y, kg_i_x / sum1)

    return kg_x

###second latest optimal version
##def calc_kg_x_WA(Y, T):
##    T_i_x = np.tile(T, (reac.Ns, 1)).transpose() / 1000  
##
##    # Viscosity of the components
##    viscosity_i_x = reac.matrixC_coef[:reac.Ns, 0] * np.power(T_i_x, reac.matrixC_coef[:reac.Ns, 1]) / (
##        np.ones_like(T_i_x) + reac.matrixC_coef[:reac.Ns, 2] + reac.matrixC_coef[:reac.Ns, 3] / T_i_x ** 2.0)
##
##    # Thermal conductivity of the components
##    kg_i_x = reac.matrixB_coef[:reac.Ns, 0] * np.power(T_i_x, reac.matrixB_coef[:reac.Ns, 1]) / (
##        np.ones_like(T_i_x) + reac.matrixB_coef[:reac.Ns, 2] / T_i_x + reac.matrixB_coef[:reac.Ns, 3] / T_i_x ** 2.0)
##
##    # Evaluate the Sutherland coefficient of species i-th
##    sum1 = np.zeros((reac.Nx, reac.Ns))
##
##    for k in range(reac.Nx):
##        YY = Y[k][:reac.Ns]
##        viscosity_ratio = np.divide.outer(viscosity_i_x[k], viscosity_i_x[k])
##        sqrt_term = np.sqrt((T_i_x[k] + reac.Si) / (T_i_x[k] + reac.Si.T))
##        value4 = (1.0 + np.sqrt(viscosity_ratio * (reac.Mj_by_Mi ** 0.75) * sqrt_term)) ** 2
##        value5 = (T_i_x[k][:, np.newaxis] + reac.Si_j) / (T_i_x[k][:, np.newaxis] + reac.Si)
##        Theta_ij = 0.25 * value4 * value5
##        sum1[k] = YY @ Theta_ij
##
##    # Evaluate the thermal conductivity of the mixture
##    kg_x = np.einsum('ij,ij->i', Y, kg_i_x / sum1)
##
##    return kg_x


# def calc_kg_x_WA(Y, T): 
#    # calculate thermal conductivity as weighted mole-frac average
   
#    T_i_x = np.tile(T,(reac.Ns,1)).transpose() # each column is T_x
#    T_i_x = T_i_x/1000 
#    # thermal conductivity of pure species i (column) at each x-location (row)
#    kg_i_x = reac.matrixB_coef[:reac.Ns,0]*np.power(T_i_x,reac.matrixB_coef[:reac.Ns,1]) / ( np.ones_like(T_i_x) + reac.matrixB_coef[:reac.Ns,2]/T_i_x + reac.matrixB_coef[:reac.Ns,3]/T_i_x**2.0 ) # verified against excel
#    kg_x = np.einsum('ij,ij->i',Y,kg_i_x) # rowwise dot product of composition and thermal conductivity
#    return kg_x
