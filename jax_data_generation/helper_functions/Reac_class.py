import jax
import jax.numpy as jnp

from Molercular_weight_i import *

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
    self.Nx = 1001 # Number of Discrete points along reactor
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
    self.DHr = 45600.0

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
