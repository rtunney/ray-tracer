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
	def __init__(self, point, normal, color='FFFFFF'):
		assert isinstance(normal, Ray), "input normal of type Ray" 
		self.point = point
		self.normal = normal
		self.color = color

	def get_intersection(self, ray):
		scale_factor = dot(self.normal.direction, (self.point - ray.point))/dot(self.normal.direction, (ray.direction))
		intersection = ray.get_coords(scale_factor)
		if intersection[2]<0: 
			return intersection, self.normal
		else: 
			return False, False

	def get_intersections(self, ray):
		scale_factor = dot(self.normal.direction, (self.point - ray.point))/dot(self.normal.direction, (ray.direction))
		intersection = ray.get_coords(scale_factor)
		if intersection[2]<0: 
			return [intersection]
		else: 
			return []

	def get_cos_theta(self, point, light, camera):
		block_point = self.get_intersection(Ray(camera.point, light.point))
		if block_point and isbetween(block_point, camera.point, light.point): 
			return 0
		else:
			light_ray = Ray(point, point2=light.point)
			cos_theta = abs(dot(light_ray.direction, self.normal.direction)
						/(linalg.norm(light_ray.direction)*linalg.norm(self.normal.direction)))
		return cos_theta

class Sphere(object):
	def __init__(self, center, radius, color='FFFFFF'):
		self.center = center
		self.radius = radius
		self.color = color

	def get_intersection(self, ray):
		a = sum((ray.point2-ray.point)**2)
		b = sum(2 * (ray.point2-ray.point) * (ray.point-self.center))
		c = sum((ray.point-self.center)**2) - self.radius**2

		test = b**2 -4*a*c

		if test>=0: 
			scale = (-b - sqrt(test)) / (2.0*a)
			intersection = ray.point + scale * (ray.direction)
			normal = Ray(self.center, intersection)
  			return intersection, normal
		else: return False, False

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

	def get_cos_theta(self, point, light, camera):
		light_ray = Ray(point, light.point)
		normal = Ray(self.center, point)
		cos_theta = abs(dot(light_ray.direction, normal.direction)
					/(linalg.norm(light_ray.direction)*linalg.norm(normal.direction)))
		return cos_theta

class Light(object):
	def __init__(self, point, color='FFFFFF'):
		self.point = point
		self.color = color

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
		normal = None
		intersected = None

		for shape in self.shapes:
			current_intersection, current_normal = shape.get_intersection(ray)
			if isinstance(current_intersection, ndarray):
				intersections.append(current_intersection)
				current_distance = linalg.norm(current_intersection-point)
				if 0.00001<current_distance<min_distance:
					min_distance = current_distance
					nearest_intersection = current_intersection
					intersected = shape
					normal = current_normal

		return nearest_intersection, intersected, normal

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

	def get_cos_theta(self, point, normal, light):
		light_ray = Ray(point, light.point)
		cos_theta = abs(dot(light_ray.direction, normal.direction)
					/(linalg.norm(light_ray.direction)*linalg.norm(normal.direction)))
		return cos_theta

	def get_lighting(self, shape, cos_theta, light):
		red = int(min(int(shape.color[:2], 16), int(light.color[:2], 16))*cos_theta)
		green = int(min(int(shape.color[2:4], 16), int(light.color[2:4], 16))*cos_theta)
		blue = int(min(int(shape.color[4:], 16), int(light.color[4:], 16))*cos_theta)
		return array([red, green, blue, 0])

	def get_pixels(self):
		pixels = []

		for point in self.screen.get_points():
			# print point
			ray = Ray(self.camera.point, point)

			nearest_intersection, shape, normal = self.get_nearest_intersection(camera.point, ray)

			if nearest_intersection==None:
				pixels.append((1, 1, 1, 255))

			else:
				illum = array([0, 0, 0, 255])
				for light in self.lights:
					if not self.isblocked(nearest_intersection, light):
						cos_theta = self.get_cos_theta(nearest_intersection, normal, light)
						illum += self.get_lighting(shape, cos_theta, light)
					#illum += sqrt(1/sqrt(get_dist(point, light.point)))
				illum += .5*(255-illum)
				pixels.append(tuple(illum))

		return pixels

	def draw(self):
		mode = 'RGBA'
		size = self.screen.res
		pixels = self.get_pixels()

		my_image = Image.new(mode, size)
		my_image.putdata(pixels)
		my_image.save('practice dontbreakplz.png')


camera = Camera(100)
screen = Screen(100, 100, (500, 500))
light0 = Light(array([100, 100, -100]))
light1 = Light(array([100, 100, -100]), color='12ECA9')
light2 = Light(array([-100, 100, -100]), color='FCF76C')
lights = [light1, light2]

p0 = array([0, 0, -100])
p1 = array([100, 0, -300])
p2 = array([-200, 200, -500])
p3 = array([0, 100, -200])

s1 = Sphere(p0, 20)
s2 = Sphere(p1, 20)
s3 = Sphere(p2, 20)
s4 = Sphere(p3, 20)

s5 = Sphere(p0, 20, color='12ECA9')
s6 = Sphere(p1, 20, color='FCF76C')
s7 = Sphere(p2, 20, color='FE3F3F')
s8 = Sphere(p3, 20, color='3706AC')

normal = Ray(array([0, 0, 0]), array([0, 1, 0]))
plane = Plane(array([0, -75, 0]), normal)

shapes = [s5, s6, s7, s8, plane]

w = World(camera, screen, lights, shapes)

w.draw()
