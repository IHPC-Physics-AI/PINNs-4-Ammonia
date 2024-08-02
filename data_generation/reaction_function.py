import time
import sys
import scipy
import pylab
import numpy as np
import matplotlib
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math

import Reactor_catalyst_design as reac
import Operational_conditions as oper

from Molecular_weight_i import*
from viscosity_mixture import*
from Cp_mixture import*
from Thermal_conductivity import*
from Pressure_drop import*
from Heat_transfer_coeffs import*
from Kinetic_model import*

P_x = oper.P0 * np.ones((reac.Nx))
Y = np.load("generated_data_no_P_drop/steady_YT_no_Pdrop_Tf_723.15_Q_1.0.npy")

def reaction_term(Y, reac, oper):

    YiT = np.reshape(Y, (reac.Nx, reac.Ns+1), order='F') # each row is molfracs at some position
    Yi_x = YiT[:,:-1] # each column is molefrac of one species
    T_x = YiT[:,-1]
    #---------------------------------------------------------
    #I. Compute the nonlinear source terms of reactions
    #---------------------------------------------------------
    F   = np.zeros(reac.N)

    # Load molecular weights
    Wi = Molecular_weight_i()[:reac.Ns]

    ######## P_x is assumed to have reached steady state
    #Set the gas pressure at the inlet
    # if(count==1):
    #     P_x = oper.P0 * np.ones((reac.Nx)) # assume no pressure drop at the first step

    MW_x= Yi_x @ Wi # molecular weight at each x-location
    # print("MW_x: ",MW_x)
    # print("T_x:", T_x)
    rhom_x = P_x / oper.R / T_x # molar density at each location
    rhog_x = MW_x  * rhom_x
    if np.any(rhog_x < 0):
        print("negative rho_g")
    Cpg_x = calc_Cpg_x(Yi_x,T_x)
    if np.any(Cpg_x < 0):
        print("negative Cpg_x")
    mug_x = calc_mug_x(Yi_x,T_x)
    if np.any(mug_x < 0):
        print("negative mug_x")
    #kg_x = calc_kg_x_WA(Yi_x,T_x, mug_x) # weighted average thermal conductivity
    kg_x = calc_kg_x_WA(Yi_x,T_x) # weighted average thermal conductivity
    if np.any(kg_x < 0):
        print("negative kg_x")
        print(YiT[np.where(kg_x < 0)][:])

    #viscosity_i_x = viscosity_i(T_x)
    #print(np.shape(viscosity_i_x))
    #print(viscosity_i_x)

    ug_x = oper.f_t * np.ones(reac.Nx) # need to change if on a mass basis
    #ug_x = 0.1*oper.ug * np.ones(reac.Nx) # need to change if on a mass basis


    ######## P_x is assumed to have reached steady state
    #Update the pressure drop
    # P_x = oper.P0 * np.ones((reac.Nx)) - pressure_drop_x(rhog_x, mug_x)*reac.xvals

    # print("kg_x: ", kg_x)
    # print("mug_x: ", mug_x)
    # print("rhog_x: ", rhog_x)
    # print("Cpg_x: ", Cpg_x)


    UP_x = calc_UP_x(kg_x,mug_x,rhog_x,ug_x,Cpg_x/MW_x)*5.0e6
    Pi_x = Yi_x * np.expand_dims(P_x, axis=-1)
    F = calc_reaction_x(T_x,Pi_x,UP_x,oper.Tf,reac.rho_b,reac.eta)
    F[reac.Ns*reac.Nx:] /= Cpg_x * rhom_x

    return F

print(reaction_term(Y, reac, oper).shape)