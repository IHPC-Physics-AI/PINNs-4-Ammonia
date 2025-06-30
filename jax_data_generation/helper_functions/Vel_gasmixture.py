import jax
import jax.numpy as jnp

from Reac_class import *
from Oper_class import *

jax.config.update("jax_enable_x64", True)

def Vel_gasmixture(reac, oper):

  SV = oper.SV0 * (oper.T0 / 273.15) / (oper.P0 / 101325) # Space Velocity at Inlet

  return SV / reac.epsilon_b

jax.jit(Vel_gasmixture, static_argnums=(0,1))
