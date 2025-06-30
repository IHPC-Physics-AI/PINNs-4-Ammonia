# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

"""# Helper Classes"""

# Helper Class: Reactor Catalyst Design
class Reac:
  def __init__(self):
    # Reactor Dims
    self.L = 1.0 # Reactor length (m)
    self.D0 = 2.54e-2 # Reactor Outer Diameter (m)
    self.Din = 2.0e-2 # Reactor Inner Diameter (m)
    self.Dp = 0.0015 # Catalyst Pellet Diameter (m)
    self.Df = self.D0 + 0.002 * 2 # Furnace / Heat Jacket Diameter (m)
    self.rho_b = 904.0 # Packing Density (kg/m**3)
    self.eta = 0.99 # Catalyst Effectiveness Factor

    # Discretizing x-axis
    self.Nx = 201 # Number of Discrete points along reactor
    self.dx = self.L /(self.Nx-1) # Size of an element (m)
    self.xvals = jnp.linspace(0,self.L,self.Nx) # Positions of Discrete Locations (m)


    '4-species Chemical Reaction Scheme'

    # Kinetic Constants
    self.Ns = 4 # Total Number of Species
    self.N = self.Nx * (self.Ns + 1) # Total Variables of the Large Matrix
    self.Nr = 1 # Number of Reactions

    # Stoichiometric Coefficients
    self.stoi_vij = jnp.array([-2.0,3.0,1.0,0.0]) # 2NH3 -> 3H2 + 1N2 (Ar Catalyst)

    # Heat Released
    self.DHr = - 45600.0

    # Species Properties (Polar / Non-polar)
    self.fs = [0.733, 1.0, 1.0, 1.0]

    # Mixture Viscoscity Subroutine
    self.matrixC_coef = jnp.array([[7.24238380e-03, 9.40613857e-01, 2.01052920e+02, 2.69408538e+00], # NH3 [0]
                 [2.65255557e-02,  -1.30542430e+00,  -9.37838615e+00, 1.28838353e+03], # H2 [1]
                 [6.89400597e-03, 6.44713803e-01, 1.63910616e+02, 1.09061213e+00], # N2 [2]
                 [8.46258883e-04, 6.45185645e-01, 1.40300927e+01, 1.91297973e-01]])

    self.Mj_by_Mi = ((jnp.asarray(Molecular_weight_i()).reshape(-1,1)/jnp.asarray(Molecular_weight_i()).reshape(1,-1)) ** 0.5)[:self.Ns, :self.Ns]

    #Builing Temperature
    self.Si = 1.5*jnp.ones(self.Ns)
    self.Tb = jnp.array([239.8, 20.3, 77.34, 87.2])
    self.fs = jnp.array([0.733, 1.0, 1.0, 1.0])
    self.Si = self.Si * self.Tb
    self.Si = self.Si.at[1].set(79)


    self.Si_j = (self.fs.reshape(-1,1) * jnp.sqrt(self.Si.reshape(1,-1) * self.Si))[:self.Ns, :self.Ns]


    self.epsilon_b = 0.38 + 0.073 * (1.0 - (self.D0/self.Dp - 2.0) ** 2.0 /(self.D0/self.Dp) ** 2.0) # Bed Porosity (0 < epsilon_b < 1)
    self.AC = jnp.pi * (self.Din/2) ** 2 # Reactor Cross Sectional Area pi * r ** 2 (m**2
    self.rho_p = self.rho_b / (1 - self.epsilon_b) # Density of Catalyst
    self.mc = self.rho_b * self.AC * self.L # Total Catalyst Weight in Reactor

    # Corrected Heat Capacity
    self.matrixA_coef = jnp.array([[19.99563, 49.77119, -15.37599, 1.921168, 0.189174], # NH3
                                  [33.066178,  -11.363417, 11.432816,  -2.772874,  -0.158558], # H2
                                  [19.505830,  19.887050,  1.3697840,   0.527601,  -4.935202], # N2
                                  [20.78600, 2.825911E-7, -1.464191E-7, 1.092131E-8, -3.661371E-8]]) # Ar

    self.matrixB_coef  = jnp.array([[0.70717629, 0.87712917, 4.09458442, -0.4458513], # NH3 [0]
                     [3.42961237e+00,  -2.90547479e-07, 7.33928075e+00,  1.93431432e-01], # H2 [1]
                     [1.79616324e+00, -2.72877800e-07, 2.70845470e+01, 3.99694386e-01], #N2 [2]
                     [1.37446554e+00, -1.20619828e-06, 3.44051496e+00, 2.81153756e+01]]) # Ar [3]


    # Diffusion matrix using fourth-order central differencing
    self.d2_dx2_indexing = jnp.array([1, self.Nx-2])

    self.d2_dx2 = jnp.zeros((self.Nx, self.Nx))
    self.d2_dx2 = self.d2_dx2.at[self.d2_dx2_indexing, self.d2_dx2_indexing -1].set(1.0 / self.dx**2)
    self.d2_dx2 = self.d2_dx2.at[self.d2_dx2_indexing, self.d2_dx2_indexing].set(-2.0 / self.dx**2)
    self.d2_dx2 = self.d2_dx2.at[self.d2_dx2_indexing, self.d2_dx2_indexing+1].set(1.0 / self.dx**2)


    self.denom4CFD = 12 * self.dx ** 2 # denominator for 4th-order CFD

    self.fourth_order_indexing = jnp.arange(2, self.Nx-2)

    self.d2_dx2 = self.d2_dx2.at[self.fourth_order_indexing, self.fourth_order_indexing-2].set(-1 / self.denom4CFD)
    self.d2_dx2 = self.d2_dx2.at[self.fourth_order_indexing, self.fourth_order_indexing-1].set(16 / self.denom4CFD)
    self.d2_dx2 = self.d2_dx2.at[self.fourth_order_indexing, self.fourth_order_indexing].set(-30 / self.denom4CFD)
    self.d2_dx2 = self.d2_dx2.at[self.fourth_order_indexing, self.fourth_order_indexing+1].set(16 / self.denom4CFD)
    self.d2_dx2 = self.d2_dx2.at[self.fourth_order_indexing, self.fourth_order_indexing+2].set(-1 / self.denom4CFD)


    #  Advection matrix using 1st order upwind (only positive flow)
    self.d_dx_indexing = jnp.arange(1, self.Nx-1)
    self.d_dx = jnp.zeros((self.Nx, self.Nx))
    self.d_dx = self.d_dx.at[self.d_dx_indexing, self.d_dx_indexing].set(1.0 / self.dx)
    self.d_dx = self.d_dx.at[self.d_dx_indexing, self.d_dx_indexing-1].set(-1.0 / self.dx)

# Helper Class: Operational Conditions

class Oper:

  def __init__(self, reac):
    self.Wi = Molecular_weight_i()
    self.R = 8.3145 # Universal Gas Constant
    self.P_std = 101325.0 # Pressure of Gas at Standard Conditions (Pa)
    self.T_std = 273.15 # Temperature of Gas at Standard Conditions (K)

    self.T0 = 303.0 # Temperature at Reactor Inlet (K)
    self.Tf = 773.0 # Temperature of the heat jacket (K)
    self.P0 = 101325.0 # Pressure at Reactor Inlet (Pa)

    self.Q_pump1 = 1.0 # Ammonia Dioxide Flowrate (kg/day)
    self.Q_pump2 = self.Q_pump1 * 0.7 / 0.3 # Argon Pump Flowrate (kg/day)

    self.f_NH3 = self.Q_pump1/self.Wi[0]/24/60/60 # Gas Molar Flow Rate of NH3 (mol/s)
    self.f_Ar = 0.0 # Gas Molar Flow Rate of Ar (mol/s)
    self.f_t = self.f_NH3 + self.f_Ar # Total Gas Molar Flow Rate

    self.Volume = (self.f_NH3 * self.Wi[0] + self.f_Ar * self.Wi[3]) * self.R * self.T_std / self.P_std # Volume of Gas in Reactor (m**3)

    self.SV0 = self.Volume / reac.AC

    self.ug = Vel_gasmixture(reac, self) # Gas Velocity along the Reactor (m/s)

    self.t_final = reac.L / self.ug # Resident time of the gas flow inside reactor (s)

    self.Da = 0.001 # Axial Mass Dispersion Term

    self.lamdaa = 0.001 # Axial Thermal Dispersion Term

"""# Helper Functions"""

# Helper Function: Molecular Weights of Reactants

def Molecular_weight_i():
  Wi = [17.0305E-3,     #N H3
        2.01588E-3,     # H2
        28.0134E-3,     # N2
        39.9480E-3]     # Ar

  return jnp.asarray(Wi)

jax.jit(Molecular_weight_i)

# Helper Function: Average Velocity of Gas Flow in Reactor
"Clarify this also"

def Vel_gasmixture(reac, oper):

  SV = oper.SV0 * (oper.T0 / 273.15) / (oper.P0 / 101325) # Space Velocity at Inlet

  return SV / reac.epsilon_b

jax.jit(Vel_gasmixture, static_argnums=(0,1))

# Helper Function: Heat Capacity of Gas Mixture via Estimated Correlation
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

# Helper Function: Viscoscity of Gas Mixture via Viscoscity of Components
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

# Helper Function: Thermal Conductivity of Gas Mixture via Estimated Correlation
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

# Helper Function: Pressure Drop along Fixed Bed Catalyst Reactor
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

# Helper Function: Overall Heat Transfer Coefficient of the Gas Mixture and Catalyst Pellets via Estimated Coefficient
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

# Helper Function: Rate of Chemical Reaction
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

def plotter(ysol):
  reac = Reac()
  oper = Oper(reac)

  # Reshape the solution to have time steps, spatial points, and species/temperature
  # ysol has shape (nSteps, reac.N) where reac.N = reac.Nx * (reac.Ns + 1)
  # We need to reshape it to (nSteps, reac.Nx, reac.Ns + 1)
  ysol_reshaped = ysol.reshape(ysol.shape[0], reac.Nx, reac.Ns + 1, order='F')

  # Separate species and temperature data
  specNH3 = ysol_reshaped[:, :, 0]
  specH2 = ysol_reshaped[:, :, 1]
  specN2 = ysol_reshaped[:, :, 2]
  specAr = ysol_reshaped[:, :, 3] # Assuming Ar is the 4th species
  reacT = ysol_reshaped[:, :, reac.Ns]

  nSteps = ysol.shape[0]

  # Create colormaps
  norm = matplotlib.colors.Normalize(vmin=0, vmax=nSteps)
  cmap1 = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.Blues)
  cmap2 = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.Oranges)
  cmap3 = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.Greens)
  cmap4 = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.Reds)
  cmapAll = [cmap1,cmap2,cmap3,cmap4]

  figsz = (10,4)

  # Plot Temperature
  plt.figure(1)
  plt.gcf().set_size_inches(figsz)
  for ti in range(0,nSteps):
    label = "Temperature" if ti == nSteps - 1 else ""
    plt.plot(reac.xvals, reacT[ti], color=cmap4.to_rgba(ti), label=label)
  ax = plt.gca()
  ax.set(xlabel="Reactor Length (m)", ylabel="Temperature (K)", title='Temp. evolution')
  ax.grid(color='gainsboro')
  ax.legend(loc='right')

  # Plot NH3
  plt.figure(2)
  plt.gcf().set_size_inches(figsz)
  for ti in range(0,nSteps):
    label = "specNH3" if ti == nSteps - 1 else ""
    plt.plot(reac.xvals, specNH3[ti], color=cmap1.to_rgba(ti), label=label)
  ax = plt.gca()
  ax.set(xlabel="Reactor Length (m)", ylabel="NH3 mole concentration", title='NH3 evolution')
  ax.grid(color='gainsboro')
  ax.legend(loc='right')

  # Plot H2
  plt.figure(3)
  plt.gcf().set_size_inches(figsz)
  for ti in range(0,nSteps):
    label = "specH2" if ti == nSteps - 1 else ""
    plt.plot(reac.xvals, specH2[ti], color=cmap2.to_rgba(ti), label=label)
  ax = plt.gca()
  ax.set(xlabel="Reactor Length (m)", ylabel="H2 mole concentration", title='specH2 evolution')
  ax.grid(color='gainsboro')
  ax.legend(loc='right')

  # Plot N2
  plt.figure(4)
  plt.gcf().set_size_inches(figsz)
  for ti in range(0,nSteps):
    label = "specN2" if ti == nSteps - 1 else ""
    plt.plot(reac.xvals, specN2[ti], color=cmap3.to_rgba(ti), label=label)
  ax = plt.gca()
  ax.set(xlabel="Reactor Length (m)", ylabel="N2 mole concentration", title='specN2 evolution')
  ax.grid(color='gainsboro')
  ax.legend(loc='right')

  # Plot Ar (Assuming you also want to plot Argon)
  plt.figure(5)
  plt.gcf().set_size_inches(figsz)
  for ti in range(0,nSteps):
      label = "specAr" if ti == nSteps - 1 else ""
      plt.plot(reac.xvals, specAr[ti], color=cmapAll[0].to_rgba(ti), label=label) # Reusing cmap1 for simplicity, you can add a new one if needed
  ax = plt.gca()
  ax.set(xlabel="Reactor Length (m)", ylabel="Ar mole concentration", title='specAr evolution')
  ax.grid(color='gainsboro')
  ax.legend(loc='right')


  plt.show()

def gen_coeffs(Y, reac, oper):

    YiT = jnp.reshape(Y, (reac.Nx, reac.Ns+1), order='F')
    Yi_x = YiT[:, :-1]
    T_x = YiT[:, -1]
    P_x = oper.P0 * jnp.ones((reac.Nx))

    Wi = Molecular_weight_i()[:reac.Ns]
    MW_x = Yi_x @ Wi
    rhom_x = oper.P0 / oper.R / T_x
    rhog_x = MW_x * rhom_x
    Cpg_x = calc_Cpg_x(Yi_x, T_x, reac)
    kg_x = calc_kg_x_WA(Yi_x, T_x, reac)
    mug_x = calc_mug_x(Yi_x, T_x, reac)

    lambda_by_Cpg = oper.lamdaa / Cpg_x

    ug_x = oper.f_t * jnp.ones(reac.Nx)
    UP_x = calc_UP_x(kg_x, mug_x, rhog_x, ug_x, Cpg_x / MW_x, reac) * 5.0e6
    Pi_x = Yi_x * jnp.expand_dims(P_x, axis=-1)

    reaction_vector = calc_reaction_x(T_x, Pi_x, UP_x, reac, oper)
    Ri = jnp.reshape(reaction_vector[:reac.Nx*reac.Ns], (reac.Nx, reac.Ns), order='F')
    RT = reaction_vector[reac.Nx*reac.Ns:] / (Cpg_x * rhom_x)

    coeffs = jnp.zeros((reac.Nx, 9))
    coeffs = coeffs.at[:, 0].set(oper.Da)  # Species diffusion
    coeffs = coeffs.at[:, 1].set(-ug_x / reac.AC)  # Species advection
    coeffs = coeffs.at[:, 2:2 + reac.Ns].set(Ri)  # Species reaction
    coeffs = coeffs.at[:, 6].set(lambda_by_Cpg)  # Temperature diffusion
    coeffs = coeffs.at[:, 7].set(-ug_x / reac.AC)  # Temperature advection
    coeffs = coeffs.at[:, 8].set(RT)  # Temperature reaction

    return coeffs
