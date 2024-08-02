#=============================================================================
# This function evaluates the bed porosity of the reactor
#=============================================================================
def Bed_porosity(D0, Dp):

    # D0 - is reactor diameter
    # Dp - is catalyst pellet diameter 

    return 0.38 + 0.073*(1.0 - (D0/Dp - 2.0)**2/(D0/Dp)**2)

