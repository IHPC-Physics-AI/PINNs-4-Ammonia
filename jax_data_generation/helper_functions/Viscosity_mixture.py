import jax
import jax.numpy as jnp

from Reac_class import *

jax.config.update("jax_enable_x64", True)

def calc_mug_x(Y, T, reac):
  """
  Y (array): Composition at each x-location
  T (array): Temperature at all x-locations
  """
  t = T[:,None] / 1000.0

  mug_i_x = reac.matrixC_coef[:reac.Ns,0] * jnp.power(t,reac.matrixC_coef[:reac.Ns,1]) / (jnp.ones_like(t) + reac.matrixC_coef[:reac.Ns,2] + reac.matrixC_coef[:reac.Ns,3]/t**2.0)
  denom = Y @ reac.Mj_by_Mi
  mug_x = jnp.einsum('ij,ij->i',Y,mug_i_x/denom)

  return mug_x

jax.jit(calc_mug_x, static_argnums=(2,))
