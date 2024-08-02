import numpy as np
import Reactor_catalyst_design as reac
import Operational_conditions as oper

def calc_reaction_x(T, P, UP, Tf, rho_b, eta):
    # Constants and parameters
    A1 = 36180.0
    A2 = 8730.0
    Ea1 = 82600.0
    Ea2 = 76710.0
    a1 = 1.5
    a2 = 0.5
    a3 = 1.0
    a41 = 0.24
    a42 = 0.28
    a51 = -0.54
    a52 = -0.42
    R = oper.R

    # Partial pressures in kPa
    P_NH3 = P[:, 0] / 1000.0
    P_H2 = P[:, 1] / 1000.0
    P_N2 = P[:, 2] / 1000.0
    P_AR = P[:, 3] / 1000.0

    # Calculate equilibrium coefficient
    log10Kp_star = -2.691122 * np.log10(T) - 5.519265E-5 * T + 1.848863E-7 * T * T + 2001.6 / T + 2.6899
    log10Kp = log10Kp_star + (0.1191849 / T + 91.87212 / np.power(T, 2) + 25122730.0 / np.power(T, 4))
    Kp = np.power(10.0, log10Kp)
    beta = Kp * np.power(P_H2, a1) * np.power(P_N2, a2) / np.power(P_NH3, a3)

    # Calculate reaction rates
    Ri_x = np.where(P_NH3 < 1.0e-6, 0, np.where(T < 748, A1 * np.exp(-Ea1 / T / R) * np.power(P_NH3, a41) * np.power(P_H2, a51) * (1.0 - np.power(beta, 2)), A2 * np.exp(-Ea2 / T / R) * (P_NH3 ** a42) * (P_H2 ** a52) * (1.0 - beta ** 2)))

    # Energy balance
    SDHr = Ri_x * reac.DHr

    # Species formation
    r_Yi_x = np.outer(Ri_x, reac.stoi_vij)

    # Concatenate arrays
    dCi = np.concatenate((r_Yi_x.ravel(order='F'), -rho_b * eta * SDHr + UP * (Tf - T)))

    return dCi
