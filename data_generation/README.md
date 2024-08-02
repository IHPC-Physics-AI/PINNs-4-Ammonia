# Data Generation

This set of code contains the Python functions to perform transient simulation of the ammonia cracking process in a 1D reactor. Different operation and reactor parameters can be set.

# Transient Simulation

The overall equation is given by the convection-diffusion of the chemicals and temperature.

``` math
\frac{\partial Y_i}{\partial t}=D_a \frac{\partial^2 Y_i}{\partial x^2} - \frac{u_g}{A_c} \frac{\partial Y_i}{\partial x} + R_i(Y,T,x) 
```
``` math
\frac{\partial T}{\partial t}=\frac{\lambda}{C_{pg}(Y,T)} \frac{\partial^2 T}{\partial x^2} - \frac{u_g}{A_c} \frac{\partial Y_i}{\partial x} + R_T(Y,T,x) 
```
The initial conditions are set using the initial concentration of the different components and initial temperature. The boundary conditions are Dirichlet at the reactor inflow and Neumann at the end of the reactor.

$$
\frac{\partial Y_i}{\partial x}(x=L)=0,  \frac{\partial T}{\partial x}(x=L)=0
$$

$$
Y_1(x=0)=0.99, Y_2(x=0)=0.005, Y_3(x=0)=0.005, T(x=0)=T_{amb}
$$

