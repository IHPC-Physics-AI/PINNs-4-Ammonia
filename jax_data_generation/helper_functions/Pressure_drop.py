import jax
import jax.numpy as jnp

from Reac_class import *
from Oper_class import *

jax.config.update("jax_enable_x64", True)

def pressure_drop_x(rho, mu, reac, oper):
  """
  rho (array): Density of the Gas Flow inside Reactor (kg/m**3)
  mu (array): Dynamic Viscosity (kg/ms**-1)
  """
  term1 = 1.75*(rho*oper.ug**2)*(1-reac.epsilon_b)/reac.Dp
  term2 = 150*mu*oper.ug*(1-reac.epsilon_b)*(1-reac.epsilon_b)/(reac.Dp**2)
  press_drop = (term1 + term2)/reac.epsilon_b**3

  return press_drop

jax.jit(pressure_drop_x, static_argnums=(2,3))
