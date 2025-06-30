import jax
import jax.numpy as jnp

def Molecular_weight_i():
  Wi = [17.0305E-3,     #N H3
        2.01588E-3,     # H2
        28.0134E-3,     # N2
        39.9480E-3]     # Ar

  return jnp.asarray(Wi)

jax.jit(Molecular_weight_i)
