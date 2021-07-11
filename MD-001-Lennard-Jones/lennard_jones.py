import os
from Utilities.Utils import SysUtils

import hoomd
import hoomd.md
import gsd.hoomd

import math
import itertools
import numpy
import matplotlib
from numpy.core.fromnumeric import repeat

# NOTE: General Sys setup
currentPath = os.path.dirname(__file__)


# Initializing the a Random System
# NOTE: Face Centered Cubic structure:
m = 4
numberOfParticles = 4 * m ** 3

# NOTE: KxKxK simple cubic lattice of width L
spacing = 1.3
K = math.ceil(numberOfParticles**(1/3))
L = K * spacing

# NOTE: In HOOMD, particle positions must be placed inside a periodic box. Cubic boxes range from -L/2 to L/2.
x = numpy.linspace(-L / 2, L / 2, K, endpoint=False)
position = list(itertools.product(x, repeat=3))
# # Debug
# print(position[0:4])

# NOTE: Filter this list down to N particles because K*K*K >= N_particles
position = position[0: numberOfParticles]

# NOTE: The quaternion (1, 0, 0, 0) represents no rotation.
orientation = [(1,0,0,0)] * numberOfParticles

# NOTE: GSD files store the periodic box, particle positions, orientations, and other properties of the state. Use the GSD Python package to write this file.
snapshot = gsd.hoomd.Snapshot()
snapshot.particles.N = numberOfParticles
snapshot.particles.position = position
snapshot.particles.orientation = orientation
# NOTE: Each particle also has a type
snapshot.particles.typeid = [0] * numberOfParticles
snapshot.particles.types = ['particleType/Shape ETC - a label']
# NOTE: Three box lengths L_x, L_y, L_z, and 3 tilt factors. Set equal box lengths 0 tilt factors to define a cubic box
snapshot.configuration.box = [L, L, L, 0, 0, 0]

# Writing the lattice to file
# NOTE: We will use the gsd to set the State
snapshotPath = SysUtils.generateSnaptshotPath(currentPath=currentPath, fileName='lattice.gsd')
with gsd.hoomd.open(name=snapshotPath, mode='xb') as file:
    file.append(snapshot)



device = hoomd.device.CPU()
simulation = hoomd.Simulation(device=device, seed = 1)
simulation.create_state_from_gsd(filename=snapshotPath)

# NOTE: Setting up the details / properties of the System
integrator = hoomd.md.Integrator(dt = 0.005)
# print(integrator.dt)

# Setting up the potential
cell = hoomd.md.nlist.Cell()
lennard_jones_Potential = hoomd.md.pair.LJ(nlist = cell)

# Pamatereizing the potential
lennard_jones_Potential.params[("A","A")] = dict(epsilon = 1, sigma = 1)
lennard_jones_Potential.r_cut[("A","A")] = 2.5

# Appending the Lennard-Jones potential to the integrator
integrator.forces.append(lennard_jones_Potential)

# Nosé-Hoover thermostat
# NOTE: Tau is a coupling constant, which i need ro read on.
nvt = hoomd.md.methods.NVT(kT = 1.5, filter = hoomd.filter.All(), tau = 1.0)
integrator.methods.append(nvt)

# NOTE: Assigning the integrator
simulation.operations.integrator = integrator

print(simulation.state)
print(simulation.operations.integrator)
print(simulation.operations.updaters[:])
print(simulation.operations.writers[:])