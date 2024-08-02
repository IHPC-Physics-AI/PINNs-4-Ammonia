#==============================================================================
# This function evaluates the viscosity of the gas component i-th
# via the correlation proposed by Perry and Green as flow
#==============================================================================


def viscosity_i(C, T):

    # C= [C1 C2 C3 C4] are the costants

    return C[0]*T**(C[1]) / (1.0 + C[2] + C[3]/T)

    
