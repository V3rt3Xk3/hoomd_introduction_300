import os
from Utilities.Utils import SysUtils, GSDUtils

import hoomd
import hoomd.md
import gsd.hoomd

import math
import itertools
import numpy
import matplotlib

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

#Debug
prositions2Print = snapshot.particles.position

snapshot.particles.orientation = orientation
# NOTE: Each particle also has a type
snapshot.particles.typeid = [0] * numberOfParticles
snapshot.particles.types = ['A']
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

# NOTE: As HoomD velocities are defaulted to 0 we have to give a starting value
# NOTE: Use the ThermodynamicQuantities class to compute properties of the system
simulation.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.5)
thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
                                                filter=hoomd.filter.All())

# NOTE: ThermodynamicQuantities is a Compute, an Operation that computes properties of the system state. Some computations can only be performed during or after a simulation run has started.
simulation.operations.computes.append(thermodynamic_properties)
simulation.run(0)

# NOTE: Compressing the System
# NOTE: ######################
# Setting the ramp
ramp = hoomd.variant.Ramp(A=0, B=1, t_start=simulation.timestep, t_ramp=20000)
rho = simulation.state.N_particles / simulation.state.box.volume

# Setting the target density
initial_box = simulation.state.box
final_box = hoomd.Box.from_box(initial_box)  # make a copy of initial_box
final_rho = 1.2
final_box.volume = simulation.state.snapshot.particles.N / final_rho

# NOTE: To avoid destabilizing the integrator with large forces due to large box size changes, scale the box with a small period.
box_resize_trigger = hoomd.trigger.Periodic(10)
box_resize = hoomd.update.BoxResize(box1=initial_box,
                                    box2=final_box,
                                    variant=ramp,
                                    trigger=box_resize_trigger)
simulation.operations.updaters.append(box_resize)

# NOTE: Running the simulation to resize the box
simulation.run(20001)
# # Debug
# print(simulation.state.box == final_box)
simulation.operations.updaters.remove(box_resize)

# WOW: This is superwierd, I couldn't find a good way to copy a hoomd.Snapshot to gsd.hoomd.Snapshot. The only difference seems like, there is a validate method for the properties and the  class itself.
gsdSnapshot = gsd.hoomd.Snapshot()
hoomdSnapshot = simulation.state.snapshot
GSDUtils.saveSnapshot("resizedBox.gsd" ,gsdSnapshot, hoomdSnapshot)

# NOTE: Running the actual simulation to resize the box
simulation.run(5e5)
# WOW: This is superwierd, I couldn't find a good way to copy a hoomd.Snapshot to gsd.hoomd.Snapshot. The only difference seems like, there is a validate method for the properties and the  class itself.
gsdSnapshot = gsd.hoomd.Snapshot()
hoomdSnapshot = simulation.state.snapshot
GSDUtils.saveSnapshot("SimulationEnd.gsd" ,gsdSnapshot, hoomdSnapshot)

print(simulation.state)
print(simulation.operations.integrator)
print(simulation.operations.updaters[:])
print(simulation.operations.writers[:])
