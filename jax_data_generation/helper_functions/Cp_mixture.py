import jax
import jax.numpy as jnp

from Reac_class import *

jax.config.update("jax_enable_x64", True)

def calc_Cpg_x(Y, T, reac):
  """
  Y (array): Composition at each x-location
  T (array): Temperature at all x-locations
  """

  t = T / 1000.0

  A = reac.matrixA_coef[:reac.Ns, :]

  Cpg_i_x = (
        A[None, :, 0] +
        A[None, :, 1] * t[:, None] +
        A[None, :, 2] * t[:, None]**2 +
        A[None, :, 3] * t[:, None]**3 +
        A[None, :, 4] / (t[:, None]**2)
  )

  Cpg_x = jnp.einsum('ij,ij->i', Y, Cpg_i_x)

  def debug_warn(Cpg_x, T, Y):
        jax.debug.print("Negative Cpg_x")
        return Cpg_x

  Cpg_x = jax.lax.cond(
        jnp.any(Cpg_x < 0),
        lambda _: debug_warn(Cpg_x, T, Y),
        lambda _: Cpg_x,
        operand=None
  )

  return Cpg_x

jax.jit(calc_Cpg_x, static_argnums=(2,))
