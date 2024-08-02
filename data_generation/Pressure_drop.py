#===============================================================================
# Function to evaluate the pressure drop along the fixed bed catalyst reactor
#===============================================================================
import numpy as np
import Reactor_catalyst_design as reac
import Operational_conditions as oper

def pressure_drop_x(rho, mu):

    # rho - density of the gas flows inside the reactor (kg/m3)
    # mu - dynamic viscosity (kg/m/s)
    
    term1 = 1.75*(rho*oper.ug**2)*(1-reac.epsilon_b)/reac.Dp
    term2 = 150*mu*oper.ug*(1-reac.epsilon_b)*(1-reac.epsilon_b)/(reac.Dp**2)
    press_drop = (term1 + term2)/reac.epsilon_b**3
    
    return press_drop
