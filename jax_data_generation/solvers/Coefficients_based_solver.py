import time
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, Kvaerno5, ODETerm, SaveAt, PIDController

from Compiled_helper_functions import *

jax.config.update("jax_enable_x64", True)

# Helper Function: Constructing 1D Convection-Diffusion-Reaction Problem
def dC_dt2(t, Y, reac, oper):

    P_x = oper.P0 * jnp.ones((reac.Nx))  # Assume no Pressure Drop at First Step


    coeffs = gen_coeffs(Y, reac, oper)

    YiT = jnp.reshape(Y, (reac.Nx, reac.Ns+1), order='F')
    Yi_x = YiT[:, :-1]
    T_x = YiT[:, -1]

    F = jnp.zeros(reac.N)
    Wi = Molecular_weight_i()[:reac.Ns]


    MW_x = Yi_x @ Wi
    rhom_x = oper.P0 / oper.R / T_x
    rhog_x = MW_x * rhom_x
    Cpg_x = calc_Cpg_x(Yi_x, T_x, reac)
    mug_x = calc_mug_x(Yi_x, T_x, reac)

    # Species reaction terms
    for spec in range(0, reac.Ns):
        F = F.at[spec*reac.Nx:(spec+1)*reac.Nx].set(coeffs[:, 2 + spec].T)
    # Temperature reaction term
    F = F.at[reac.Ns*reac.Nx:].set(coeffs[:, 8].T)

    # --- Diffusion ---
    Y_diff = jnp.zeros_like(Y)
    for spec in range(0, reac.Ns):
        d2Y_i_dx2 = reac.d2_dx2 @ Y[spec*reac.Nx:spec*reac.Nx + reac.Nx]
        Y_diff = Y_diff.at[spec*reac.Nx:spec*reac.Nx + reac.Nx].set(coeffs[:, 0].T * d2Y_i_dx2)
    # Temperature diffusion
    d2T_dx2 = reac.d2_dx2 @ Y[reac.Ns*reac.Nx:reac.Ns*reac.Nx + reac.Nx]
    Y_diff = Y_diff.at[reac.Ns*reac.Nx:reac.Ns*reac.Nx + reac.Nx].set(coeffs[:, 6].T * d2T_dx2)

    # --- Advection ---
    Y_adv = jnp.zeros_like(Y)
    adv_vel = oper.f_t / reac.AC
    # adv_vel = 0.0

    # Individual Species Advection and Temperature
    # All have the Same Advection Term (? to confirm)
    for spec in range(0,reac.Ns+1):
        dY_i_dx = reac.d_dx @ Y[spec*reac.Nx:spec*reac.Nx + reac.Nx]
        Y_adv = Y_adv.at[spec*reac.Nx:spec*reac.Nx + reac.Nx].set(- adv_vel* dY_i_dx)


    # --- Boundary Conditions ---
    Y_BC = jnp.zeros_like(Y)
    adv_vel = oper.f_t / reac.AC
    for spec in range(0, reac.Ns):
        C_BC_out = jnp.zeros((reac.Nx))
        beta_out = 0
        b_out = 2 * beta_out * oper.Da / reac.dx
        C_BC_out = C_BC_out.at[-1].set(-oper.Da * 2 / reac.dx**2)
        C_BC_out = C_BC_out.at[-1].add(-adv_vel / reac.dx)
        C_BC_out = C_BC_out.at[-2].set(oper.Da * 2 / reac.dx**2)
        C_BC_out = C_BC_out.at[-2].add(adv_vel / reac.dx)
        Y_BC = Y_BC.at[(spec+1)*reac.Nx - 1].set(jnp.dot(C_BC_out, Y[spec*reac.Nx:spec*reac.Nx + reac.Nx]) + b_out)

    # Temperature Neumann BC
    spec = reac.Ns
    C_BC_out = jnp.zeros((reac.Nx))
    beta_out = 0
    dTdt_coeff = rhom_x * Cpg_x
    b_out = 2 * beta_out * oper.lamdaa / reac.dx / dTdt_coeff[-1]
    C_BC_out = C_BC_out.at[-1].set(-oper.lamdaa * 2 / reac.dx**2 / dTdt_coeff[-1])
    C_BC_out = C_BC_out.at[-2].set(oper.lamdaa * 2 / reac.dx**2 / dTdt_coeff[-2])
    C_BC_out = C_BC_out.at[-1].add(-adv_vel / reac.dx)
    C_BC_out = C_BC_out.at[-2].add(adv_vel / reac.dx)
    Y_BC = Y_BC.at[(spec+1)*reac.Nx - 1].set(jnp.dot(C_BC_out, Y[spec*reac.Nx:spec*reac.Nx + reac.Nx]) + b_out)

    AY = Y_diff + Y_adv + Y_BC

    # --- Control vector ---
    Bu = jnp.zeros(reac.N)

    dYdt = AY + Bu + F
    dYdt = dYdt.at[::reac.Nx].set(0.0) # at x = 0, dYdt = 0 for fixed-value inlet boundary for all species


    return dYdt
jax.jit(dC_dt2, static_argnums=(2,3))

def main2():
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

  Y0 = Y

  def ode_func_diffrax(t, y, args):
      reac_arg, oper_arg = args
      dYdt_result = dC_dt2(t, y, reac_arg, oper_arg)
      return dYdt_result
  jax.jit(ode_func_diffrax)

  # Time span
  t0 = 0.0
  t1 = 60.0

  # Arguments for the ODE function (P_x_val, reac, oper)
  args = (reac, oper)

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

jax.jit(main2)

if __name__ == "__main__":
  ysol = main2()
  plotter(ysol)
