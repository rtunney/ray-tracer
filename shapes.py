from __future__ import division
from numpy import ndarray, array, linalg, dot, arange, pi, sin, cos, tan, sqrt
from PIL import Image

def isbetween(point, bound1, bound2):
	# print point, bound1, bound2
	if (bound1[0]<point[0]<bound2[0] or bound1[0]>point[0]>bound2[0] and 
		bound1[1]<point[1]<bound2[1] or bound1[1]>point[1]>bound2[1] and
		bound1[2]<point[2]<bound2[2] or bound1[2]>point[2]>bound2[2]):
		return True
	else: return False

def get_dist(point1, point2):
	return linalg.norm(point2-point1)

class Ray(object):
	def __init__ (self, point, point2):
		self.point = point
		self.point2 = point2
		self.direction = point2-point 

	def get_coords(self, scale):
		return self.point + scale * self.direction

	def get_rot_matrix(self, dim, theta):
		if dim == 'x': return [[1, 0, 0], [0, cos(theta), sin(theta)], [0, -sin(theta), cos(theta)]]
		elif dim == 'y': return [[cos(theta), 0, -sin(theta)], [0, 1, 0], [sin(theta), 0, cos(theta)]]
		else: return [[cos(theta), sin(theta), 0], [-sin(theta), cos(theta), 0], [0, 0, 1]]

	def rotate(self, dim, theta):
		'''input: dim 'x', 'y', 'z' or tuple of dims,
		theta in rads or tuple of thetas
		completes rotations in order entered'''
		theta = tuple(theta); dim = tuple(dim);
		for cur_theta, cur_dim in zip(theta, dim):
			assert isinstance(cur_theta, int) or isinstance(cur_theta, float), "Theta must be a number"
			assert cur_dim in 'xyz', "Dimension must be x, y, or z"
			self.direction = dot(self.get_rot_matrix(cur_dim, cur_theta), self.direction)

class Plane(object):
	def __init__(self, point, normal):
		assert isinstance(normal, Ray), "input normal of type Ray" 
		self.point = point
		self.normal = normal

	def get_intersection(self, ray):
		scale_factor = dot(self.normal.direction, (self.point - ray.point))/dot(self.normal.direction, (ray.direction))
		intersection = ray.get_coords(scale_factor)
		if intersection[2]<0: 
			return intersection
		else: 
			return False

	def get_intersections(self, ray):
		scale_factor = dot(self.normal.direction, (self.point - ray.point))/dot(self.normal.direction, (ray.direction))
		intersection = ray.get_coords(scale_factor)
		if intersection[2]<0: 
			return [intersection]
		else: 
			return []

	def get_lighting(self, point, light, camera):
		block_point = self.get_intersection(Ray(camera.point, light.point))
		if block_point and isbetween(block_point, camera.point, light.point): 
			return 0
		else:
			light_ray = Ray(point, point2=light.point)
			cos_theta = abs(dot(light_ray.direction, self.normal.direction)
						/(linalg.norm(light_ray.direction)*linalg.norm(self.normal.direction)))
			illum = cos_theta
		return illum

class Sphere(object):
	def __init__(self, center, radius):
		self.center = center
		self.radius = radius

	def get_intersection(self, ray):
		a = sum((ray.point2-ray.point)**2)
		b = sum(2 * (ray.point2-ray.point) * (ray.point-self.center))
		c = sum((ray.point-self.center)**2) - self.radius**2

		test = b**2 -4*a*c

		if test>=0: 
			scale = (-b - sqrt(test)) / (2.0*a);
  			return ray.point + scale * (ray.direction);
		else: return False

	def get_intersections(self, ray):
		intersections = []

		a = sum((ray.point2-ray.point)**2)
		b = sum(2 * (ray.point2-ray.point) * (ray.point-self.center))
		c = sum((ray.point-self.center)**2) - self.radius**2

		test = b**2 -4*a*c

		if test>=0: 
			scale = (-b - sqrt(test)) / (2.0*a)
  			intersections.append(ray.point + scale * (ray.direction))
  			scale = (-b + sqrt(test)) / (2.0*a)
  			intersections.append(ray.point + scale * (ray.direction))
  			return intersections
		else: return []

	def get_lighting(self, point, light, camera):
		light_ray = Ray(point, light.point)
		normal = Ray(self.center, point)
		cos_theta = abs(dot(light_ray.direction, normal.direction)
					/(linalg.norm(light_ray.direction)*linalg.norm(normal.direction)))
		illum = cos_theta
		return illum

class Light(object):
	def __init__(self, point):
		self.point = point

class Camera(object):
	def __init__(self, distance):
		self.point = array([0, 0, distance])

class Screen(object):
	def __init__(self, width, height, res):
		self.width = width
		self.height = height
		self.res = res

	def get_points(self):
		x_range = arange(self.res[0])*self.width/self.res[0] - self.width/2
		y_range = (arange(self.res[1])*self.height/self.res[1] - self.height/2)[::-1]

		for y in y_range:
			for x in x_range:
				yield array([x, y, 0])

class View(object):
	def __init__(self, cam_dist, scr_width, scr_height, res):
		self.camera = Camera(cam_dist)
		self.screen = Screen(scr_width, scr_height, res)

class World(object):
	def __init__(self, camera, screen, lights, shapes):
		'input: Camera instance, screen instance, list of lights, list of shapes'
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
		intersection, shape, normal = intersection_data
		if isinstance(intersection, bool): return -inf
		else: return intersection.coords[2]

	def get_nearest_intersection(self, point, ray):
		intersections = []

		min_distance = float('inf')
		nearest_intersection = None
		intersected = None

		for shape in self.shapes:
			current_intersection = shape.get_intersection(ray)
			if isinstance(current_intersection, ndarray):
				intersections.append(current_intersection)
				current_distance = linalg.norm(current_intersection-point)
				if 0.00001<current_distance<min_distance:
					min_distance = current_distance
					nearest_intersection = current_intersection
					intersected = shape

		return min_distance, nearest_intersection, intersected, intersections

	def get_intersections(self, ray):
		intersections = []
		for shape in self.shapes:
			intersections += shape.get_intersections(ray)
		return intersections

	def isblocked(self, point, light):
		ray = Ray(point, light.point)
		intersections = self.get_intersections(ray)
		for intersection in intersections:
			if get_dist(point, intersection)>.01 and isbetween(intersection, point, light.point):
				return True
		return False

	def get_pixels(self):
		pixels = []

		for point in self.screen.get_points():
			# print point
			ray = Ray(self.camera.point, point)

			z_index, nearest_intersection, shape, intersections = self.get_nearest_intersection(camera.point, ray)

			if nearest_intersection==None:
				pixels.append((1, 1, 1, 1))

			else:
				illum = 0
				for light in self.lights:
					if not self.isblocked(nearest_intersection, light):
						illum += shape.get_lighting(nearest_intersection, light, self.camera)
					#illum += sqrt(1/sqrt(get_dist(point, light.point)))
				if illum>=1: illum=1
				illum += .5*(1-illum)
				illum = int(illum*256)
				pixels.append((illum, illum, illum, 256))

		return pixels

	def draw(self):
		mode = 'RGBA'
		size = self.screen.res
		pixels = self.get_pixels()

		my_image = Image.new(mode, size)
		my_image.putdata(pixels)
		my_image.save('practice8.png')


camera = Camera(100)
screen = Screen(100, 100, (500, 500))
light = Light(array([100, 100, -100]))
lights = [light]

p0 = array([0, 0, -100])
p1 = array([100, 0, -300])
p2 = array([-200, 200, -500])
p3 = array([0, 100, -200])

s1 = Sphere(p0, 20)
s2 = Sphere(p1, 20)
s3 = Sphere(p2, 20)
s4 = Sphere(p3, 20)

normal = Ray(array([0, 0, 0]), array([0, 1, 0]))
plane = Plane(array([0, -75, 0]), normal)

shapes = [s1, s2, s3, s4, plane]

w = World(camera, screen, lights, shapes)

w.draw()