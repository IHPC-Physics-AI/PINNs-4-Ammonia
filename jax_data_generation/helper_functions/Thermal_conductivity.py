import jax
import jax.numpy as jnp

from Reac_class import *

def calc_kg_x_WA(Y, T, reac):
  """
  Y (array): Composition at each x-location
  T (array): Temperature at all x-locations
  """

  t = jnp.tile(T[:, None], (1, reac.Ns)) / 1000.0  # Nx x Ns

  # Viscosity of Components
  viscosity_i_x = reac.matrixC_coef[:reac.Ns,0] * jnp.power(t,reac.matrixC_coef[:reac.Ns,1]) / (jnp.ones_like(t) + reac.matrixC_coef[:reac.Ns,2] + reac.matrixC_coef[:reac.Ns,3]/t**2.0)

  # Thermal Conductivity of Components
  kg_i_x = reac.matrixB_coef[:reac.Ns,0] * jnp.power(t, reac.matrixB_coef[:reac.Ns,1]) / (jnp.ones_like(t) + reac.matrixB_coef[:reac.Ns,2]/t + reac.matrixB_coef[:reac.Ns,3]/t**2.0)

  # Evaluate the Sutherland Coefficient of Species i-th
  def sum1_computer(Yk, Tk, viscosityk):
    viscosity_ratio = viscosityk[:,None] / viscosityk[None,:]
    sqrt_term = jnp.sqrt((Tk + reac.Si) / (Tk + reac.Si.T))
    value4 = (1.0 + jnp.sqrt(viscosity_ratio * (reac.Mj_by_Mi ** 0.75) * sqrt_term)) ** 2
    value5 = (Tk + reac.Si_j) / (Tk + reac.Si)
    Theta_ij = 0.25 * value4 * value5
    return jnp.einsum('i,ij->j', Yk, Theta_ij)

  compute_sum1_all = jax.vmap(sum1_computer, in_axes=(0, 0, 0))
  sum1 = compute_sum1_all(Y, t, viscosity_i_x)

  kg_x = jnp.einsum('ij,ij->i', Y, kg_i_x / sum1)

  return kg_x

jax.jit(calc_kg_x_WA, static_argnums=(2,))
