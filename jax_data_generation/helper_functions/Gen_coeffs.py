import jax
import jax.numpy as jnp

from Reac_class import *
from Oper_class import *
from Cp_mixture import *
from Heat_transfer_coeffs import *
from Kinetic_model import *
from Molecular_weight_i import *
from Thermal_conductivity import *
from Viscosity_mixture import *

jax.config.update("jax_enable_x64", True)

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

jax.jit(gen_coeffs, static_argnums(1,2))
