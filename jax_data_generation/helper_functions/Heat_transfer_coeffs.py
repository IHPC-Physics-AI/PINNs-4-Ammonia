import jax
import jax.numpy as jnp

from Reac_class import *

jax.config.update("jax_enable_x64", True)

def calc_UP_x(k, mu, rho, vel, Cp, reac):
  """
  k (array): Thermal Conductivity of Gas Mixture at all x-locations
  mu (array): Dynamic Viscosity of Gas Mixture at all x-locations
  rho (array): Density of Gas Mixture at all x-locations
  vel (array): Velocity of Gas Mixture at all x-locations
  Cp (array): Specific Heat Capacity of Gas Mixture at all x-locations
  """
  Re = rho*vel*reac.Dp/mu
  Pr = Cp*mu/k
  ho = (k/reac.Dp)*(2.58*pow(Re,1/3)*pow(Pr,1./3.) + 0.094*pow(Re,0.8)*pow(Pr,0.4))
  UP_x = 1.0 / (jnp.log(reac.Df/reac.D0)/2/jnp.pi/k + 1.0/ho/jnp.pi/reac.D0)
  return UP_x

jax.jit(calc_UP_x, static_argnums=(5))
