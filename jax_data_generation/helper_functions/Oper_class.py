import jax
import jax.numpy as jnp

from Reac_class import *

class Oper:

  def __init__(self, reac):
    self.Wi = Molecular_weight_i()
    self.R = 8.3145 # Universal Gas Constant
    self.P_std = 101325.0 # Pressure of Gas at Standard Conditions (Pa)
    self.T_std = 273.15 # Temperature of Gas at Standard Conditions (K)

    self.T0 = 303.0 # Temperature at Reactor Inlet (K)
    self.Tf = 773.0 # Temperature of the heat jacket (K)
    self.P0 = 101325.0 # Pressure at Reactor Inlet (Pa)

    self.Q_pump1 = 1.0 # Ammonia Dioxide Flowrate (kg/day)
    self.Q_pump2 = self.Q_pump1 * 0.7 / 0.3 # Argon Pump Flowrate (kg/day)

    self.f_NH3 = self.Q_pump1/self.Wi[0]/24/60/60 # Gas Molar Flow Rate of NH3 (mol/s)
    self.f_Ar = 0.0 # Gas Molar Flow Rate of Ar (mol/s)
    self.f_t = self.f_NH3 + self.f_Ar # Total Gas Molar Flow Rate

    self.Volume = (self.f_NH3 * self.Wi[0] + self.f_Ar * self.Wi[3]) * self.R * self.T_std / self.P_std # Volume of Gas in Reactor (m**3)

    self.SV0 = self.Volume / reac.AC

    self.ug = Vel_gasmixture(reac, self) # Gas Velocity along the Reactor (m/s)

    self.t_final = reac.L / self.ug # Resident time of the gas flow inside reactor (s)

    self.Da = 0.001 # Axial Mass Dispersion Term

    self.lamdaa = 0.001 # Axial Thermal Dispersion Term
