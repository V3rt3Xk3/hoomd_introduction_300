import os
import math
import numpy

import fresnel
import PIL
import freud
import gsd.hoomd
import matplotlib

from Utilities.Utils import SysUtils

# NOTE: General Sys setup
currentPath = os.path.dirname(__file__)
fileName = "resizedBox"
filePath = SysUtils.generateSnaptshotPath(currentPath=currentPath, fileName=(fileName + ".gsd"))





with gsd.hoomd.open(filePath) as trajectoryFile:
    # NOTE: Here we could use a for loop and the snapshot visualizer function, then merge them to a gif
    print(len(trajectoryFile))
    snapshot = trajectoryFile[0]


box = snapshot.configuration.box
numberOfParticles = snapshot.particles.N

particleTypes = snapshot.particles.typeid
# # Debug
# print(particleTypes)

# NOTE: Coloring the particles.
colors = numpy.empty((numberOfParticles,3))
# HACK: Shorthand for setting the value of the given index if the same indexed element in particleTypes is 0
colors[particleTypes == 0] = fresnel.color.linear([0.2, 0.2, 0.95]) # 0 Type

# NOTE: fresnel Setup
scene = fresnel.Scene()

# NOTE: Spheres for every particle in the system
geometry = fresnel.geometry.Sphere(scene, N=numberOfParticles, radius=0.2)
geometry.position[:] = snapshot.particles.position
geometry.material = fresnel.material.Material(roughness=0.9)
geometry.outline_width = 0.05

# NOTE: use color instead of material.color
geometry.material.primitive_color_mix = 1.0
geometry.color[:] = fresnel.color.linear(colors)

# NOTE: create box in fresnel - box_radius is the radius of the edges of the box
fresnel.geometry.Box(scene, box, box_radius=0.02, box_color=[1,1,1])

# NOTE: Now that we have everything setup, we will render everything 
scene.camera = fresnel.camera.Orthographic.fit(scene)
scene.lights = fresnel.light.lightbox()
fresnel.pathtrace(scene, light_samples=5)

out = fresnel.preview(scene=scene)
print(out[:].shape)
print(out[:].dtype)

image = PIL.Image.fromarray(out[:], mode='RGBA')
filePath = SysUtils.generateSnaptshotPath(currentPath=currentPath, fileName=(fileName + '.png'))
image.save(filePath)

image = PIL.Image.fromarray(out[:,:,0:3], mode='RGB')
filePath = SysUtils.generateSnaptshotPath(currentPath=currentPath, fileName=(fileName + '.jpg'))
image.save(filePath)



   