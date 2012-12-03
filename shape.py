from __future__ import division
from numpy import ndarray, array, linalg, dot, arange, pi, sin, cos, tan, inf
from PIL import Image

class Point (object):
	def __init__(self, np_array):
		assert isinstance(np_array, ndarray)
		assert len(np_array)==3
		self.coords = np_array

	def __str__(self):
		return "(" + str(self.coords[0]) + ", " + str(self.coords[1]) + ", " + str(self.coords[2]) + ")"

	def __repr__(self):
		return "array([" + str(self.coords[0]) + ", " + str(self.coords[1]) + ", " + str(self.coords[2]) + "])"

class Ray(Point):
	def __init__(self, np_array):
		super(Ray, self).__init__(np_array)
		self.length = linalg.norm(self.coords)

	def scale(self, new_len):
		self.coords = self.coords*(new_len/self.length)
		self.length = linalg.norm(self.coords)

	def get_rot_matrix (self, dim, theta):
		if dim == 'x': return [[1, 0, 0], [0, cos(theta), sin(theta)], [0, -sin(theta), cos(theta)]]
		elif dim == 'y': return [[cos(theta), 0, -sin(theta)], [0, 1, 0], [sin(theta), 0, cos(theta)]]
		else: return [[cos(theta), sin(theta), 0], [-sin(theta), cos(theta), 0], [0, 0, 1]]

	def rotate (self, dim, theta):
		'''input: dim 'x', 'y', 'z' or tuple of dims,
		theta in rads or tuple of thetas
		completes rotations in order entered'''
		theta = tuple(theta); dim = tuple(dim);
		for cur_theta, cur_dim in zip(theta, dim):
			assert isinstance(cur_theta, int) or isinstance(cur_theta, float), "Theta must be a number"
			assert cur_dim in 'xyz', "Dimension must be x, y, or z"
			self.coords = dot(self.get_rot_matrix(cur_dim, cur_theta), self.coords)
       
class Plane(object):
	def __init__(self, normal, point):
		assert isinstance(normal, Ray), "input normal of type Ray" 
		assert isinstance(point, Point), "input point of type Point"
		self.point = point
		self.normal = normal

	def get_intersection(self, ray):
		scale_factor = dot(self.point.coords, self.normal.coords)/dot(ray.coords, self.normal.coords)
		intersection = Point(ray.coords*scale_factor)
		if -1000<intersection.coords[2]<0: 
			return intersection, self.normal
		else: 
			return False

	def get_lighting(self, point, camera, lights):
		illum = 0
		for light in lights:
			intersection = self.get_intersection(Ray())

class Sphere(object):
	def __init__(self, center, radius):
		assert isinstance(center, Point)
		assert isinstance(radius, int) or isinstance(radius, point)
		self.center = center
		self.radius = radius

	def get_intersection(self, ray):
		zeroth_coef = sum(self.center.coords**2)-(self.radius**2)
		oneth_coef = -2 * dot(self.center.coords, ray.coords)
		twoth_coef = (ray.coords[0]/ray.coords[2])**2 + (ray.coords[1]/ray.coords[2])**2 + ray.coords[2]**2
		nearest_z = max(roots([twoth_coef, oneth_coef, zeroth_coef]))
		if instanceof(nearest_z, complex128): 
			return False
		else: 
			intersection = Point(ray.coords*(nearest_z/ray.coords[2]))
			return intersection, Ray(intersection.coords-self.center)

class Block(object):
	def __init__(self, point, width, height, depth):
		assert isinstance(point, Point), "input anchor point of type Point"
		self.point = point 
		self.rays = [Ray(array([width, 0, 0])), Ray(array([0, height, 0])), Ray(array([0, 0, -depth]))]
		self.planes = self.get_planes()
		self.bounds = self.get_bounds()

	def get_planes(self):
		planes = []
		for ray in self.rays:
			planes.append(Plane(ray, self.point))
			planes.append(Plane(ray, Point(self.point.coords + ray.coords)))
		return planes

	def get_bounds(self):
		counterpoint = self.point.coords + self.rays[0].coords + self.rays[1].coords + self.rays[2].coords
		return zip(self.point.coords, counterpoint)

	def get_intersection(self, ray):
		intersections = []
		for plane in self.planes:
			intersection = plane.get_intersection(ray)
			# if isinstance(intersection, tuple):
			# 	for axis, ray in enumerate(self.rays):
			# 		filter(lambda ray: ray!=plane.normal, self.rays)
			intersections.append(intersection)
		intersections = filter(lambda x: isinstance(x[0], ndarray), intersections)
		intersections = filter(lambda x: self.bounds[0][0]<=x[0][0]<=self.bounds[0][1] and 
			self.bounds[1][0]<=x[0][1]<=self.bounds[1][1] and self.bounds[2][0]>=x[0][2]>=self.bounds[2][1], intersections)

		if intersections==[]: return False
		else: return max(intersections, key=lambda x: x[0][2])

	def rotate(self, dim, theta):
		'''input: dim 'x', 'y', 'z' or tuple of dims,
		theta in rads or tuple of thetas
		completes rotations in order entered'''
		for ray in self.rays:
			ray.rotate(dim, theta)
		self.planes = self.get_planes
		self.bounds = self.get_bounds

class Disk(object):
	def __init__ (self, point, radius, depth):
		assert isinstance(point, Point)
		self.point = point
		self.radius = radius
		self.ray = Ray(array([0, 0, -depth]))

class Light(Point):
	def __init__(self, np_array):
		super(Light, self).__init__(np_array)

class Screen(object):
	def __init__(self, width, height, depth, res):
		assert isinstance(width, int) or isinstance(width, float), "input numerical width"
		assert isinstance(height, int) or isinstance(height, float), "input numerical height"
		assert isinstance(depth, int) or isinstance(depth, float), "input numerical depth"
		assert isinstance(res, tuple) and isinstance(res[0], int) and isinstance(res[1], int), "input res=(int, int)"
		self.width = width
		self.height = height
		self.depth = depth
		self.res = res

	def get_points(self):
		# for i in range(self.res[0]):
		# 	for j in range(self.res[1]):
		# 		x = -self.width/2 + self.width*(i/self.res[0])
		# 		y = -self.height/2 + self.height*(j/self.res[1])
		# 		yield array([x, y, 0])
		x_range = arange(self.res[0])*self.width/self.res[0] - self.width/2
		y_range = arange(self.res[1])*self.height/self.res[1] - self.height/2
		for x in x_range:
			for y in y_range:
				yield Point(array([x, y, self.depth]))
		# yield array([x, y, 0]) for y in y_range for x in x_range

class Camera(Point):
	def __init__(self):
		self.coords = array([0, 0, 0])

class World(object):
	def __init__(self, camera, screen, lights, shapes):
		'input: Camera instance, list of lights, list of shapes'
		self.camera = camera
		self.screen = screen
		self.lights = lights
		self.shapes = shapes

	def set_screen(self, screen):
		self.screen = screen

	def change_resolution(self, res):
		self.screen.res = res

	def add_light(self, light):
		self.lights.append(light)

	def remove_light(self, index):
		del self.lights[index]

	def add_shape(self, shape):
		self.shapes.append(shape)

	def remove_shape(self, index):
		del self.shapes[index]

	def nearest_intersect(intersection_data):
		intersection, normal = intersection_data
		if isinstance(intersection, bool): return -inf
		else: return intersection.coords[2]

	def get_pixels(self):
		pixels = []

		for point in self.screen.get_points():
			ray = Ray(point.coords)
			intersections = []

			for shape in self.shapes:
				intersection = shape.get_intersection(ray)
				if intersection: intersections.append(intersection)

			if intersections == []:
				pixels.append((0, 0, 0, 1))
				continue

			else:
				intersection, normal = max(intersections, key=self.nearest_intersect)

				illum = 0
				for light in self.lights:
					illum += shape.get_lighting(intersection, normal)
				if illum>=1: illum=1
				illum = int(illum*256)
				pixels.append((illum, illum, illum, 256))

		return pixels

	def draw(self):
		mode = 'RGBA'
		size = self.screen.res
		pixels = self.get_pixels()

		my_image = Image.new(mode, size)
		my_image.putdata(pixels)
		my_image.save('practice.png')


p = Point(array([1, 1, 1]))
n = Ray(array([0, 0, 1]))
r = Ray(array([6, 8, 2]))
pl = Plane(n, p)

p2 = Point(array([0, 0, -3]))
r2 = Ray(array([1, 0, 1]))
s = Sphere(p2, 1)

screen = Screen(10, 10, 3, (10, 10))

p3 = Point(array([-2, -2, -5]))
b = Block(p3, 5, 5, 5)

p4 = Point(array([5, 5, -5]))
b2 = Block(p4, 5, 5, 5)

r3 = Ray(array([0, 0, -1]))
