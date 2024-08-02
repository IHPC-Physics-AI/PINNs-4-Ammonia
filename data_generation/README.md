# Data Generation

This set of code contains the Python functions to perform transient simulation of the ammonia cracking process in a 1D reactor. Different operation and reactor parameters can be set.

The code for data generation can be found in ```data_generation.ipynb```. The code involves defining an initial condition ```Y0``` for the values (4 reactants (Y) + temperature (T) at 51 sample points) and a function for the rate of change ```dC_dt(t,Y, reac, oper)```, which is then passed into ```solve_ivp(dC_dt,[0, tvals[-1]],Y0,args=(reac, oper), t_eval=tvals, method='LSODA')``` to solve for the values at different times.

A function to compute the reaction term of the equation is found in ```reaction_function.py```. The function to compute $C_{pg}$ is found in ```Cp_mixture.py```.

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

# Intermediate Quantities

## Heat Capacity $C_{pg}$

```Cpg_x = calc_Cpg_x(Yi_x,T_x)```

$$ \begin{align}
C_{pg}(x)&=\sum_{i=1}^{4}Y_i(x) C_{pg,i}(x), \\
C_{pg,i}(x)&=A0_i+A1_i\times T(x)+A2_i\times T(x)^2 + A3_i\times T(x)^3 + A4_i \times T(x)^{-2}
\end{align} $$

## Viscosity $\mu_g$

```mug_x = calc_mug_x(Yi_x,T_x)```

$$ \begin{align}
\mu_g(x) &= \sum_{i=1}^4 \frac{Y_i(x) \mu_{g,i}(x)}{\sum_{j=1}^{4}Y_j(x) \sqrt{\frac{Mw_j(x)}{Mw_i(x)}}}\\
\mu_{g,i}(x)&=\frac{C0_i T(x)^{C1_i}}{1+C2_i+\frac{C3_i}{T(x)^2}}
\end{align} $$

## Thermal Conductivity $K_g$

```kg_x = calc_kg_x_WA(Yi_x,T_x, mug_x)```$\in \mathbf{R}^{101\times 1}$


$$ \begin{align}
K_{g,i}(x)&=\frac{B0_i T(x)^{B1_i}}{1+\frac{B2_i}{T(x)}+\frac{B3_i}{T(x)^2}}\\
K_g(x) &= \sum_{i=1}^4 Y_i(x) K_{g,i}(x)
\end{align}$$

$$\begin{align}
\theta_{ij}=\frac{1}{4} \left[1+\left[\left(\frac{\mu_{g,i}}{\mu_{g,j}}\right)\left(\frac{Mw_j}{Mw_i}\right)^{3/4}\left(\frac{T+S_i}{T+S_j}\right) \right]^{\frac{1}{2}}\right]^{2} \times \frac{T+S_{ij}}{T+S_{i}}
\end{align}$$

## Heat Transfer Coefficient $UP$

```UP_x = calc_UP_x(kg_x,mug_x,rhog_x,ug_x,Cpg_x/MW_x)*5.0e6```

$$
u_g(x)=0.00226536, D_p=0.0015\\
$$

$$ \begin{align}
Re(x)&=\frac{\rho_g(x) u_g D_p}{\mu_g(x)}\\
Pr(x)&=\frac{C_{pg}(x)}{Mw(x) }\frac{\mu_g(x)}{K_g(x)}\\
h_o(x)&=\frac{K_g(x)}{D_p} \left(2.58Re(x)^{1/3}Pr(x)^{1/3}+0.094Re(x)^{0.8}Pr(x)^{0.4}\right)
\end{align}$$

$$
UP(x)=\left(\frac{ln\left(\frac{D_f}{D_o}\right)}{2\pi K_g(x)}+\frac{1}{h_o(x) \pi D_o}\right) \times 5 \times 10^{6}
$$

# Reaction Term $R$

If $P_1(x)<10^{-6}$,

$$
R(x)=0
$$

Else:

$$
K_P^*(x) = -2.691122 log_{10} (T(x)) - 5.519265\times 10^{-5}T(x) + 1.848863\times 10^{-7}T(x)^2 + \frac{2001.6}{T(x)} + 2.6899\\
$$

$P_t = 0.1$

$$
K_P(x)=K_P^*(x) + (\frac{0.1191849}{T(x)} + \frac{91.87212}{T(x)^2} + \frac{25122730.0}{T(x)^4})\times P_t\\
K_{eq}(x)=10^{K_P(x)}
$$

$$
\beta=\frac{1}{K_{eq}(x)} \frac{P_{2}(x)^{a_1} P_{3}(x)^{a_2}}{P_{1}(x)^{a_3}}
$$

If $T(x)<748$:

$$
R(x)=A1\times e^{-\frac{Ea_1}{RT(x)}} \times P_{1}(x)^{a_{41}} \times P_{2}(x)^{a_{51}} \times (1-\beta^2)
$$

Else:

$$
R(x)=A2\times e^{-\frac{Ea_2}{RT(x)}} \times P_{1}(x)^{a_{42}} \times P_{2}(x)^{a_{52}} \times (1-\beta^2)
$$

$s_{1}=2, s_{2}=-3, s_{3}=-1, s_{4}=0$

$$
R_i(x) = R(x) \times s_i(x)
$$

$DHr=-45600.0, \rho_b=904.0, \eta=0.99, T_f=673.0$

$$
SDHr+=R(x)\times DHr\\
R_T(x)=-\frac{\rho_b \times \eta \times SDHr + UP(x)\times(T_f-T(x))}{C_{pg}(x) \times \rho_m (x)}
$$
