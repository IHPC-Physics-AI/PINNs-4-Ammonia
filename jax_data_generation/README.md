# Data Generation with JAX
## Introduction
This set of code contains the functions found in the ```data_generation``` folder rewritten with ```JAX```. As much of the functions are conceptually the same, this ```README``` will focus more on the differences between this and the original code.

The rest of the ```README``` will re-introduce the Ammonia Cracking Simulation and its governing PDEs, followed by briefing exploring the differences between these functions and their ```NumPy``` counterparts, lastly elaborating on the ```Diffrax``` solvers.

## Transient Simulation

The overall equation is given by the convection-diffusion of the chemicals and temperature.

``` math
\frac{\partial Y_i}{\partial t}=D_a \frac{\partial^2 Y_i}{\partial x^2} - \frac{u_g}{A_c} \frac{\partial Y_i}{\partial x} + R_i(Y,T,x) 
```
``` math
\frac{\partial T}{\partial t}=\frac{\lambda}{C_{pg}(Y,T)} \frac{\partial^2 T}{\partial x^2} - \frac{u_g}{A_c} \frac{\partial T}{\partial x} + R_T(Y,T,x) 
```
The initial conditions are set using the initial concentration of the different components and initial temperature. The boundary conditions are Dirichlet at the reactor inflow and Neumann at the end of the reactor.

$$
\frac{\partial Y_i}{\partial x}(x=L)=0,  \frac{\partial T}{\partial x}(x=L)=0
$$

$$
Y_1(x=0)=0.99, Y_2(x=0)=0.005, Y_3(x=0)=0.005, T(x=0)=771.5K \left(\text{arbitrary number chosen}\right)
$$

## Key Differences
One of the most important differences between these scripts and the original ones in ```data_generation``` is that the sign of enthalpy change was flipped. This changes the reaction from exothermic to endothermic and corrects the error in the original script.

With that being said, the error is not noticeable in the original scripts as the extremely low BC of $$T_{amb}$$ restricts sufficient magnification to notice the flawed shape of the temperature plot.

## Differences in Helper Functions
The primary difference between the helper functions found here and those found in ```data_generation``` is the existence of two new helper functions in this folder, ```Gen_coeffs``` and ```plotter```.

### Plotter
As the latter was made purely for convenience and does not deviate significantly from the code used in ```data_generation``` for plotting, the helper function ```plotter```, along with the other helper functions, will not be further elaborated on.

### Gen_coeffs
On the other hand, ```Gen_coeffs``` exists as a convenient way to calculate the coefficients to the Transient Simulation PDEs at any given time.

It accepts the molar concentrations of all 4 species and temperature concantated together as an array ```Y0``` as the input, along with class objects storing reaction and operational conditions ```reac``` amd ```oper```. It then outputs all 9 unique coefficients, as shown in the table below.

| Output| Corresponding Coefficient|
|---|---|
|coeffs[:,0]| Diffusion constant $$D_a$$ for all species as shown in PDE 1|
|coeffs[:,1]| Advection constant $$\frac{u_g}{A_c}$$ for all species as shown in PDE 1|
|coeffs[:,2]| Reaction term $$R_i(Y,T,x)$$ for $$NH_3$$ as shown in PDE 1|
|coeffs[:,3]| Reaction term $$R_i(Y,T,x)$$ for $$H_2$$ as shown in PDE 1|
|coeffs[:,4]| Reaction term $$R_i(Y,T,x)$$ for $$N_2$$ as shown in PDE 1|
|coeffs[:,5]| Reaction term $$R_i(Y,T,x)$$ for $$Ar$$ as shown in PDE 1|
|coeffs[:,6]| Diffusion term $$\frac{\lambda}{C_{pg}(Y,T)}$$ for $$T$$ as shown in PDE 2|
|coeffs[:,7]| Advection constant $$\frac{u_g}{A_c}$$ for $$T$$ as shown in PDE 2|
|coeffs[:,8]| Reaction term $$R_T(Y,T,x)$$ for $$T$$ as shown in PDE 2|

As shown above, each column corresponds to a certain coefficient, while each row corresponds to a position $$x$$.

## Solvers
This folder contains two solvers, ```Original_solver``` and ```Coefficients_based_solver```. Both of them utilise the ```Diffrax``` module. All three components will be further elaborated on below.

### Diffrax
```Diffrax``` is a numerical ODE/PDE solver compatible with ```JAX```. Both solvers utilised the ```Kvaerno5``` method, and the time span was set to be between $$t_0 = 0.0s$$ to $$t_1 = 60.0s$$. Both solvers had 201 spatial points across a reactor with length $$L = 1.0m$$. Both solvers returned a total of 201 y values across the integration time.

### Original_solver
```Original_solver``` was created to follow the original ```data_generation``` scripts as closely as possible. This was done to validate the helper functions transcribed over from ```data_generation```.

### Coefficients_based_solver
```Coefficients_based_solver``` was created to re-express the PDE in terms of its coefficients and to solve it based on that. This was done to validate the helper function ```gen_coeffs```.
