# Multi Variable PINNs

## Introduction

This set of code consists PINNs that only solve for the quantity of both Temperature or NH3 Concentration across a set of spatial points within the reactor at steady state. The Molar Concentrations of the other species of gases can be obtained via stoichiometry. The PINN was only trained on a single case (as described in ```jax_data_generation```), which corresponds to the ground truth vector ```steady_state_solution.npy```.

## Organisation of Folder
Currently, this folder only contains the linear PINN.

## Network Architecture
As mentioned in the parent folder, the structure of the PINNs consists of several hidden layers feeding into a pseudoinverse.

Half of the nodes of the final hidden layer will be assigned to the derivation of NH3 Concentration, while the other half will be assigned to the derivation of Temperature. Within the pseudoinverse matrix $A$, the upper half of the matrix $A$ corresponds to NH3 Concentrations at the spatial coordinates, while the bottom half corresponds to Temperature at the spatial coordinates.

Since the nodes are divided for both variables, within the upper half of the matrix, elements in the matrix corresponding to Temperature will be set to zero, while elements in the matrix corresponding to NH3 Concentration in the lower half of the matrix will be set to zero.

Lastly, the pseudoinverse is solved linearly — the ground truth vector ```steady_state_solution.npy``` will be fed into the ```gen_coeffs``` function to derive the correct coefficients. Thus, the pseudoinverse will only be implemented once per training iteration.
