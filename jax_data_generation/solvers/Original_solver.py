pip install diffrax

import time
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, Kvaerno5, ODETerm, SaveAt, PIDController

from Compiled_helper functions import *

jax.config.update("jax_enable_x64", True)

# Helper Function: Constructing 1D Convection-Diffusion-Reaction Problem
def dC_dt(t,Y, P_x_input, reac, oper):

  def problem():
    #jax.debug.print("y is negative")
    #jax.debug.print('y = {}', Y)
    return jnp.clip(Y, 1e-3, None)


  Y = jax.lax.cond(jnp.any(Y < 0), lambda: problem(), lambda: Y)

  YiT = jnp.reshape(Y, (reac.Nx, reac.Ns+1), order='F') # Each Row is Molfracs at some Position
  Yi_x = YiT[:,:-1] # Each Column is Molfrac of One Species
  T_x = YiT[:,-1]

  #---------------------------------------------------------
  #I. Compute the Nonlinear Source Terms of Reactions
  #---------------------------------------------------------
  F = jnp.zeros(reac.N)

  # Load Molecular Weights
  Wi = Molecular_weight_i()[:reac.Ns]

  P_x = P_x_input # Pressure at each x-position

  MW_x= Yi_x @ Wi # Molecular Weight at each x-position
  rhom_x = P_x / oper.R / T_x # Molar Density at each Location
  rhog_x = MW_x  * rhom_x
  jax.lax.cond(jnp.any(rhog_x < 0), lambda: jax.debug.print("negative rho_g"), lambda: None)
  Cpg_x = calc_Cpg_x(Yi_x,T_x, reac)
  jax.lax.cond(jnp.any(Cpg_x < 0), lambda: jax.debug.print("negative Cpg_x"), lambda: None)
  mug_x = calc_mug_x(Yi_x,T_x, reac)
  jax.lax.cond(jnp.any(mug_x < 0), lambda: jax.debug.print("negative mug_x"), lambda: None)
  kg_x = calc_kg_x_WA(Yi_x,T_x, reac) # Weighted Average Thermal Conductivity
  jax.lax.cond(jnp.any(kg_x < 0), lambda: jax.debug.print("negative kg_x"), lambda: None)

  ug_x = oper.f_t * jnp.ones(reac.Nx) # Need to Change if on a Mass Basis

  # Update the Pressure Drop
  P_x = oper.P0 * jnp.ones((reac.Nx)) - pressure_drop_x(rhog_x, mug_x, reac, oper)*reac.xvals


  UP_x = calc_UP_x(kg_x,mug_x,rhog_x,ug_x,Cpg_x/MW_x, reac)*5.0e6
  Pi_x = Yi_x * jnp.expand_dims(P_x, axis=-1)
  jax.lax.cond(jnp.any(Pi_x < 0), lambda: jax.debug.print("negative Pi_x"), lambda: None)
  F = calc_reaction_x(T_x,Pi_x,UP_x, reac, oper)
  F = F.at[reac.Ns*reac.Nx:].set(F[reac.Ns*reac.Nx:] / (Cpg_x * rhom_x))

  #---------------------------------------------------------
  #II. Form the RHS Matrix A = Convection + Diffusion
  #---------------------------------------------------------

  # Calculate Y_adv and Y_diff Componentwise

  Y_diff = jnp.zeros_like(Y)

  # Individual Species Diffusion
  d2Y_dx2_all = reac.d2_dx2 @ YiT
  species_diff_terms = oper.Da * d2Y_dx2_all[:, :reac.Ns]
  temp_diff_term = oper.lamdaa * d2Y_dx2_all[:, reac.Ns] / Cpg_x
  Y_diff = jnp.concatenate((species_diff_terms.flatten(order='F'), temp_diff_term.flatten(order='F')))

  Y_adv = jnp.zeros_like(Y)
  adv_vel = oper.f_t/reac.AC
  # adv_vel = 0.0

  # Individual Species Advection and Temperature
  # All have the Same Advection Term (? to confirm)
  dY_dx_all = reac.d_dx @ YiT
  Y_adv_all = -adv_vel * dY_dx_all
  Y_adv = Y_adv_all.flatten(order='F')


  # Y_BC is needed
  Y_BC = jnp.zeros_like(Y) # with no further modifications, BCs are fixed at initial value
  C_BC_out_species = jnp.zeros((reac.Nx))
  C_BC_out_species = C_BC_out_species.at[-1].set(-oper.Da*2/reac.dx**2 - adv_vel / reac.dx)
  C_BC_out_species = C_BC_out_species.at[-2].set(oper.Da*2/reac.dx**2 + adv_vel / reac.dx)

  C_BC_out_temp = jnp.zeros((reac.Nx))
  dTdt_coeff = rhom_x*Cpg_x
  C_BC_out_temp = C_BC_out_temp.at[-1].set(-oper.lamdaa*2/reac.dx**2 / dTdt_coeff[-1] - adv_vel / reac.dx)
  C_BC_out_temp = C_BC_out_temp.at[-2].set(oper.lamdaa*2/reac.dx**2 / dTdt_coeff[-2] + adv_vel / reac.dx)

  def calculate_bc_for_column(y_col, bc_coeffs):
      return jnp.dot(bc_coeffs, y_col)

  species_bc_terms = jax.vmap(calculate_bc_for_column, in_axes=(1, None))(Yi_x, C_BC_out_species)
  temp_bc_term = calculate_bc_for_column(T_x, C_BC_out_temp)

  bc_indices = jnp.arange(reac.Nx - 1, reac.N, reac.Nx)
  all_bc_values = jnp.concatenate((species_bc_terms, jnp.array([temp_bc_term])))
  Y_BC = Y_BC.at[bc_indices].set(all_bc_values)


  AY = Y_diff + Y_adv + Y_BC

  #---------------------------------------------------------
  #III. Build the control vector B*u(t)
  #---------------------------------------------------------
  Bu = jnp.zeros(reac.N)

  # Right hand-sided functions dCi/dt = A*Ci(t) + B*u(t) + F(Ci,t)
  dYdt = AY + Bu + F
  dYdt = dYdt.at[::reac.Nx].set(0.0) # at x = 0, dYdt = 0 for fixed-value inlet boundary for all species

  return dYdt, P_x

jax.jit(dC_dt, static_argnums=(3,4))

def main():
  # Instantiate Reac and Oper classes
  reac = Reac()
  oper = Oper(reac)

  # Initial Conditions (as in the original notebook)
  initial_mol_frac_NH3 = 1.0
  f_H2_initial = 0.0
  f_N2_initial = 0.0
  f_Ar_initial = 0.0

  initial_mol_frac_H2 = f_H2_initial / oper.f_t
  initial_mol_frac_N2 = f_N2_initial / oper.f_t
  initial_mol_frac_Ar = f_Ar_initial / oper.f_t

  Y = jnp.zeros(reac.N)
  for i in range(0,reac.Nx):
    # Set initial temperature for all spatial points
    Y = Y.at[reac.Ns*reac.Nx + i].set(oper.T0)

    # Set initial species concentrations for all spatial points
    Y = Y.at[i].set(oper.f_NH3/oper.f_t - 0.005)
    Y = Y.at[reac.Nx+i].set(0.0025)
    Y = Y.at[2*reac.Nx+i].set(0.0025)
    Y = Y.at[3*reac.Nx+i].set(0.0) #oper.f_AR/oper.f_t - 0.005


  # Change inlet temperature boundary condition to 771.5K
  Y = Y.at[reac.Ns * reac.Nx].set(771.5)


  T_initial = jnp.zeros(reac.Nx)
  T_initial = T_initial.at[0].set(oper.T0)

  Y0 = Y

  def ode_func_diffrax(t, y, args):
      P_x_val, reac_arg, oper_arg = args
      dYdt_result, _ = dC_dt(t, y, P_x_val, reac_arg, oper_arg)
      return dYdt_result
  jax.jit(ode_func_diffrax)

  # Time span
  t0 = 0.0
  t1 = 60.0

  # Initial pressure values for the pressure drop calculation within dC_dt.
  P_x_initial_for_ode = oper.P0 * jnp.ones(reac.Nx)

  # Arguments for the ODE function (P_x_val, reac, oper)
  args = (P_x_initial_for_ode, reac, oper)

  # Define the ODE term
  term = ODETerm(ode_func_diffrax)

  # Choose a solver (Tsit5 is a good general-purpose choice, similar to RK45)
  solver = Kvaerno5()


  # Initial step size (important for diffrax)
  # A reasonable initial step size for spatial integration
  dt0 = t1 / 1000.0 # Start with a fraction of the spatial grid step

  NT = 201

  # Define when to save the solution
  saveat = SaveAt(ts = jnp.linspace(t0, t1, NT+1))

  # Set up the step size controller with relative and absolute tolerances
  stepsize_controller =PIDController(rtol=1e-3, atol=1e-3)

  # Increase Max_steps
  max_steps = 8192
  #print("Launching Diffrax Solver...")
  start_time = time.time()
  # Solve the ODE using diffrax
  sol_diffrax = diffeqsolve(
      term,
      solver,
      t0,
      t1,
      dt0,
      y0=Y0,
      args=args,
      saveat=saveat,
      stepsize_controller=stepsize_controller,
      max_steps=max_steps
  )

  # Extract results from the solution object
  Y_solution_diffrax = sol_diffrax.ys # This contains the solution at t_eval points

  return Y_solution_diffrax

jax.jit(main)

if __name__ == "__main__":
  ysol = main()
  plotter(ysol)
