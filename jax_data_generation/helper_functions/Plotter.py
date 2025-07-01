import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from Reac_class import *
from Oper_class import *

jax.config.update("jax_enable_x64", True)

def plotter(ysol):
  reac = Reac()
  oper = Oper(reac)

  # Reshape the solution to have time steps, spatial points, and species/temperature
  # ysol has shape (nSteps, reac.N) where reac.N = reac.Nx * (reac.Ns + 1)
  # We need to reshape it to (nSteps, reac.Nx, reac.Ns + 1)
  ysol_reshaped = ysol.reshape(ysol.shape[0], reac.Nx, reac.Ns + 1, order='F')

  # Separate species and temperature data
  specNH3 = ysol_reshaped[:, :, 0]
  specH2 = ysol_reshaped[:, :, 1]
  specN2 = ysol_reshaped[:, :, 2]
  specAr = ysol_reshaped[:, :, 3] # Assuming Ar is the 4th species
  reacT = ysol_reshaped[:, :, reac.Ns]

  nSteps = ysol.shape[0]

  # Create colormaps
  norm = matplotlib.colors.Normalize(vmin=0, vmax=nSteps)
  cmap1 = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.Blues)
  cmap2 = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.Oranges)
  cmap3 = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.Greens)
  cmap4 = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.Reds)
  cmapAll = [cmap1,cmap2,cmap3,cmap4]

  figsz = (10,4)

  # Plot Temperature
  plt.figure(1)
  plt.gcf().set_size_inches(figsz)
  for ti in range(0,nSteps):
    label = "Temperature" if ti == nSteps - 1 else ""
    plt.plot(reac.xvals, reacT[ti], color=cmap4.to_rgba(ti), label=label)
  ax = plt.gca()
  ax.set(xlabel="Reactor Length (m)", ylabel="Temperature (K)", title='Temp. evolution')
  ax.grid(color='gainsboro')
  ax.legend(loc='right')

  # Plot NH3
  plt.figure(2)
  plt.gcf().set_size_inches(figsz)
  for ti in range(0,nSteps):
    label = "specNH3" if ti == nSteps - 1 else ""
    plt.plot(reac.xvals, specNH3[ti], color=cmap1.to_rgba(ti), label=label)
  ax = plt.gca()
  ax.set(xlabel="Reactor Length (m)", ylabel="NH3 mole concentration", title='NH3 evolution')
  ax.grid(color='gainsboro')
  ax.legend(loc='right')

  # Plot H2
  plt.figure(3)
  plt.gcf().set_size_inches(figsz)
  for ti in range(0,nSteps):
    label = "specH2" if ti == nSteps - 1 else ""
    plt.plot(reac.xvals, specH2[ti], color=cmap2.to_rgba(ti), label=label)
  ax = plt.gca()
  ax.set(xlabel="Reactor Length (m)", ylabel="H2 mole concentration", title='specH2 evolution')
  ax.grid(color='gainsboro')
  ax.legend(loc='right')

  # Plot N2
  plt.figure(4)
  plt.gcf().set_size_inches(figsz)
  for ti in range(0,nSteps):
    label = "specN2" if ti == nSteps - 1 else ""
    plt.plot(reac.xvals, specN2[ti], color=cmap3.to_rgba(ti), label=label)
  ax = plt.gca()
  ax.set(xlabel="Reactor Length (m)", ylabel="N2 mole concentration", title='specN2 evolution')
  ax.grid(color='gainsboro')
  ax.legend(loc='right')

  # Plot Ar (Assuming you also want to plot Argon)
  plt.figure(5)
  plt.gcf().set_size_inches(figsz)
  for ti in range(0,nSteps):
      label = "specAr" if ti == nSteps - 1 else ""
      plt.plot(reac.xvals, specAr[ti], color=cmapAll[0].to_rgba(ti), label=label) # Reusing cmap1 for simplicity, you can add a new one if needed
  ax = plt.gca()
  ax.set(xlabel="Reactor Length (m)", ylabel="Ar mole concentration", title='specAr evolution')
  ax.grid(color='gainsboro')
  ax.legend(loc='right')


  plt.show()
