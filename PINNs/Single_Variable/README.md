# 1 Variable PINNs

## Introduction

This set of code consists PINNs that only solve for the quantity of a single variable (Temperature or NH3 Concentration) across a set of spatial points within the reactor at steady state. The PINN was only trained on a single case (as described in ```jax_data_generation```), which corresponds to the ground truth vector ```steady_state_solution.npy```.

## Organisation of Folder
This folder can be further divided into three segments:
- PINN that solves exclusively for Temperature
- PINN that solves exclusively for NH3 Concentration

As the network architecture of both PINNs are similar, they will be addressed together in the following segment.

## Network Architecture
As mentioned in the parent folder, the structure of the PINNs consists of several hidden layers feeding into a pseudoinverse. All the nodes of the final hidden layer will be assigned to the derivation of the variable relevant to the PINN. Lastly, the pseudoinverse is solved linearly — the ground truth vector ```steady_state_solution.npy``` will be fed into the ```gen_coeffs``` function to derive the correct coefficients. Thus, the pseudoinverse will only be implemented once per training iteration.
