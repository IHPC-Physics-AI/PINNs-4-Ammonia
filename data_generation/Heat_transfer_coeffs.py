#==============================================================================
# This function evaluates the overall heat transfer coefficient of the gas mixture
# and the catalyst pellets via the estimated coefficient 
#==============================================================================
import numpy as np
from Molecular_weight_i import*
import Reactor_catalyst_design as reac

def calc_UP_x(k, mu, rho, vel, Cp):
    Re = rho*vel*reac.Dp/mu
    Pr = Cp*mu/k
    ho = (k/reac.Dp)*(2.58*pow(Re,1/3)*pow(Pr,1./3.) + 0.094*pow(Re,0.8)*pow(Pr,0.4))
    # print(ho)
    UP_x = 1.0 / (np.log(reac.Df/reac.D0)/2/np.pi/k + 1.0/ho/np.pi/reac.D0)
    return UP_x
