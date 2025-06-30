import jax
import jax.numpy as jnp

from Reac_class import *
from Oper_class import *

jax.config.update("jax_enable_x64", True)

def calc_reaction_x(T, P, UP, reac, oper):

  """
  T (Float): Temperature at a x-location
  P (Array): Partial Pressures of Different Species at a x-location
  UP (Float: Overall Heat Transfer Parameter
  """

  Ns = reac.Ns
  Nr = reac.Nr

  #Assign Parameters
  Pt = 1.0
  A1 = 36180.0
  A2 = 8730.0
  Ea1 = 82600
  Ea2 = 76710

  a1  = 1.5
  a2  = 0.5
  a3  = 1.0

  a41 = 0.24
  a42 = 0.28
  a51 = -0.54
  a52 = -0.42


  stoi_vij = reac.stoi_vij
  DHr = reac.DHr
  R = oper.R
  Tf = oper.Tf
  rho_b = reac.rho_b
  eta = reac.eta

  #Partial Pressure
  P_NH3 = P[:,0]/1000.0
  P_H2  = P[:,1]/1000.0
  P_N2 = P[:,2]/1000.0
  P_AR = P[:,3]/1000.0

  Kr_x = jnp.zeros(reac.Nx)
  Ri_x = jnp.zeros(reac.Nx)
  r_Yi_x = jnp.zeros((reac.Nx,Ns))
  SDHr  = jnp.zeros(reac.Nx)

  # Calculate Equilibrium Coefficient
  log10Kp_star = -2.691122 * jnp.log10(T) - 5.519265E-5 * T + 1.848863E-7 * T * T + 2001.6 / T + 2.6899
  log10Kp = log10Kp_star + (0.1191849 / T + 91.87212 / jnp.power(T, 2) + 25122730.0 / jnp.power(T, 4))
  Kp = jnp.power(10.0, log10Kp)
  beta = Kp * jnp.power(P_H2, a1) * jnp.power(P_N2, a2) / jnp.power(P_NH3, a3)

  # Calculate Reaction Rates
  Ri_x = jnp.where(P_NH3 < 1.0e-6, 0,
                  jnp.where(T < 748, A1 * jnp.exp(-Ea1 / T / R) * jnp.power(P_NH3, a41) * jnp.power(P_H2, a51) * (1.0 - jnp.power(beta, 2)),
                            A2 * jnp.exp(-Ea2 / T / R) * (P_NH3 ** a42) * (P_H2 ** a52) * (1.0 - beta ** 2)))

  # Energy Balance
  SDHr  = SDHr + Ri_x*DHr

  # For each Reaction, Add the Species Formation
  r_Yi_x += jnp.outer(Ri_x,stoi_vij[:])

  dCi_part1 = jnp.reshape(r_Yi_x, (reac.Ns*reac.Nx),order='F')

  dCi_part2 = -rho_b * eta * SDHr + UP*(Tf-T)

  dCi = jnp.concatenate([dCi_part1, dCi_part2])

  return jnp.asarray(dCi)

jax.jit(calc_reaction_x, static_argnums=(3,4))
