Ray-Tracer
==========

This repo contains shapes.py, a basic ray tracing algorithm, along with several example images charting the development of the algorithm in test-images

======================
World Class
======================

•To generate an image, you must first create a world object
•A world can be thought of as a three dimensional grid with shapes, lights, a camera, and a screen
•Points on this grid are represented as numpy arrays of length 3
•To generate an image, call worldname.draw(filename). Filename is a string including a file extension that python image library supports for rgba inputs (.png is a good choice)

===============================
Cameras, Screens and Lights
===============================

•A camera is a fixed point recessed from the screen by an input distance
•A screen is an object of input width/height in the xy plane at z=0
•The resolution of the screen determines the number of pixels generated in the image produced
•A screen is centered with respect to the camera (shortest path from camera to screen passes through center of screen)

=================
Shapes
=================

•Currently worlds can contain planes, spheres, and blocks of arbitrary dimensions. 
•Class Ray supports the implementation of these shapes. 

•Rays are defined by two points. The first point is fixed.
•Planes are defined by a fixed point and a normal ray
•Spheres are defined by a central point and a radius
•Blocks are defined by a fixed 'anchor point' and three rays 

•Planes, spheres and blocks can have colors input as six character RGB hex values (e.g. 'A176C2')
•Rays, planes, and blocks can be rotated about their fixed points

=================
The Algorithm
=================

•world.get_points generates a list of points in the screen based on the dimensions and resolution of the screen
•For each point, world.get_pixels creates a ray from the camera through the point.
•world.get_nearest_intersection finds the first intersection between the ray and a shape in the field
•world.get_lighting finds the summed light coming to the intersection point from all lights in the field 
•lighting is ignored from a given light if there are any shapes between that light and the intersection point
•world.get_ambient_lighting gets ambient light for the intersection point based on the lights in the field 
•world.get_pixels returns lighting for all points as a list of RGBA tuples
•world.draw generates the image using the list of pixels and python image library

•NB: The lighting scheme accounts for RGB values on an individual basis. Illumination reflects the minimum of the elementwise RGB values of the shape and the light, scaled down by the angle of incidence. 

=======================
Areas for Improvement
=======================
•Speed
•Add reflections/refractions/transparency
•Add additional shape classes (polygonal meshes?)
•Improve illumination of dark side of objects