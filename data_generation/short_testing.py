# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 15:20:39 2024

@author: neobs1
"""
import numpy as np
import Reactor_catalyst_design as reac
import Operational_conditions as oper

import Kinetic_model
import Heat_transfer_coeffs

Tx = np.array([400,450,500,550,600])
Yi_x = np.array([[0.5, 0.5],
       [0.6, 0.4],
       [0.7, 0.3],
       [0.8, 0.2],
       [0.9, 0.1]])

Tx = 400*np.ones(reac.Nx)
Yi_x = 0.5*np.ones((reac.Nx,2))

Pi_x = oper.P0*Yi_x

print(Kinetic_model.calc_reaction_x(Tx,Pi_x,Tx,Tx,Tx,Tx))
print(Heat_transfer_coeffs.calc_UP_x(Tx,Tx,Tx,Tx,Tx))