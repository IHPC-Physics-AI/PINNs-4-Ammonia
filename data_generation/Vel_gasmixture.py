#==============================================================================
# This subrountine is develop to calculate for the velocity of the gas flow
# inside the reactor. This average velocity is calculated based on the space
# velocity at the inlet, temperature, pressure, catalyst weight (mc), and porosity
#==============================================================================

def Vel_gasmixture(T0, P0, SV0, D0, Dp, mc):

    # Initial pressure:         P0 [Pa]
    # Initial gas temperature:  T0 [K]
    # Space velocity at the standard T0 and P0: SV0 [m/s]
    # Reactor internal diameter:    D0 [m]
    # Catalyst pellet diameter :    Dp [m]
    # Catalyst weight          :    mc [kg] 
    

    # Space velocity at the inlet
    SV = SV0*(T0/273.15)*(101325.0/P0)

    #print(SV)

    # The bed porosity is calculated as
    epsilon_b = 0.38 + 0.073*(1.0 - ((D0/Dp - 2.0)**2)/(D0/Dp)**2)


    # Cross-section area of the reactor
    Ac = 3.1425*(D0/2)**2

    # Average velocity of the Gas flow inside the reactor

    #return mc*SV/Ac/epsilon_b
    return SV/epsilon_b

