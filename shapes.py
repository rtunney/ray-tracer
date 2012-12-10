from __future__ import division
from numpy import ndarray, array, linalg, dot, arange, pi, sin, cos, sqrt, inf, column_stack, newaxis, minimum, append, where
from PIL import Image

def isbetween(point, bound1, bound2):
    '''input: 3 iterables of length 3
    returns True if point is between bounds elementwise and inclusively, else False
    '''
    if (bound1[0]<=point[0]<=bound2[0] or bound1[0]>=point[0]>=bound2[0] and 
        bound1[1]<=point[1]<=bound2[1] or bound1[1]>=point[1]>=bound2[1] and
        bound1[2]<=point[2]<=bound2[2] or bound1[2]>=point[2]>=bound2[2]):
        return True
    else: return False

def get_dist(point1, point2):
    '''input: 2 numpy arrays of equal length
    returns euclidean distance between points 
    '''
    return linalg.norm(point2-point1)

#contains common functions for shapes and lights
class Shape(object):
    
    #converts a shape's self.color hex color value to 
    #an [r, g, b, 0] numpy array on the 255 scale
    def get_rgb(self):
            red = int(self.color[:2], 16)
            green = int(self.color[2:4], 16)
            blue = int(self.color[4:], 16)
            return array([red, green, blue, 0])

class Ray(object):
    #models rays as 2 points
    #stores difference of points as convenient direction attribute
    def __init__ (self, point, point2):
        self.point = point
        self.point2 = point2
        self.direction = point2-point 

    def __str__(self):
        return 'point: ' + str(self.point) + ' direction: ' + str(self.direction)

    def get_coords(self, scale):
        '''input: scalar multiplier for ray.direction
        returns a point that many ray.directions from the base point
        '''
        return self.point + scale * self.direction

    #returns a matrix that rotates rays clockwise through
    #the given angle in radians around the given axis. 
    def get_rot_matrix(self, dim, theta):
        if dim == 'x': 
            return [[1, 0, 0], [0, cos(theta), sin(theta)], [0, -sin(theta), cos(theta)]]
        elif dim == 'y': 
            return [[cos(theta), 0, -sin(theta)], [0, 1, 0], [sin(theta), 0, cos(theta)]]
        elif dim == 'z': 
            return [[cos(theta), sin(theta), 0], [-sin(theta), cos(theta), 0], [0, 0, 1]]
        else: raise ValueError, "dimension must be 'x', 'y' or 'z'"

    def rotate(self, dim, theta):
        '''input: list of dims 'x', 'y', and/or 'z',
        list of thetas in rads
        completes clockwise rotations in order entered'''
        for cur_dim, cur_theta in zip(dim, theta):
            self.direction = dot(self.get_rot_matrix(cur_dim, cur_theta), self.direction)
            self.point2 = self.point + self.direction

class Plane(Shape):
    def __init__(self, point, normal, color='FFFFFF'):
        assert isinstance(normal, Ray), "input normal of type Ray" 
        self.point = point
        self.normal = normal
        self.color = color
        self.rgb = self.get_rgb()

    def __str__(self):
        return 'point: ' + str(self.point) + ' normal ' + str(self.normal.direction)

    #get visible intersections between plane and ray
    def get_intersection(self, ray):
        scale_factor = (dot(self.normal.direction, (self.point - ray.point)) / 
                       dot(self.normal.direction, (ray.direction)))
        intersection = ray.get_coords(scale_factor)
        if intersection[2]<0: 
            return intersection, self.normal
        else: 
            return False, False

    def get_intersections(self, ray):
        scale_factor = (dot(self.normal.direction, (self.point - ray.point)) / 
                       dot(self.normal.direction, (ray.direction)))
        intersection = ray.get_coords(scale_factor)
        if intersection[2]<0: 
            return [intersection]
        else: 
            return []

class Sphere(Shape):
    def __init__(self, center, radius, color='FFFFFF'):
        self.center = center
        self.radius = radius
        self.color = color
        self.rgb = self.get_rgb()

    def __str__(self):
        return 'center ' + str(self.center) + ' radius ' + str(self.radius)

    #get nearest intersection between sphere and ray
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

    #get all intersections between sphere and ray
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

class Block(Shape):
    def __init__(self, point, width, height, depth, color='FFFFFF'):
        self.point = point
        self.rays = [Ray(self.point, self.point + array([width, 0, 0])), 
                    Ray(self.point, self.point + array([0, height, 0])), 
                    Ray(self.point, self.point + array([0, 0, -depth]))]
        self.planes = self.get_planes()
        self.color = color
        self.rgb = self.get_rgb()

    def __str__(self):
        return ('anchor ' + str(self.point) + ' l x w x h = ' + 
                str((self.rays[0][0], self.rays[1][1], -self.rays[2][2])))

    def get_planes(self):
        planes = []
        for ray in self.rays:
            planes.append(Plane(self.point, ray))
            planes.append(Plane(self.point + ray.direction, ray))
        return planes

    def rotate(self, dim, theta):
        '''input: list of dims 'x', 'y', and/or 'z',
        list of thetas in rads
        completes rotations in order entered'''
        for ray in self.rays:
            ray.rotate(dim, theta)
        self.planes = self.get_planes()

    def get_scalar_multipliers(self, point):
        '''returns an array of scalar multipliers for 
        self.rays to sum to the given point
        '''
        m = column_stack((self.rays[0].direction[:,newaxis], 
            self.rays[1].direction[:,newaxis], self.rays[2].direction[:,newaxis]))
        return linalg.solve(m, point)

    def in_face(self, point, plane, plane_index):
        #returns True if point is in plane, else returns False
        norm_index = plane_index//2
        coeffs = self.get_scalar_multipliers(point-plane.point)
        for i in xrange(3):
            if i!=norm_index:
                if not 0<=coeffs[i]<=1:
                    return False
        return True

    #get foremost intersection with a ray
    def get_intersection(self, ray):
        nearest_intersection = array([0, 0, -inf])
        normal = None
        for pl_index, plane in enumerate(self.planes): 
            current_intersection, current_normal = plane.get_intersection(ray)
            if not self.in_face(current_intersection, plane, pl_index):
                current_intersection = False; current_normal = False
            if (isinstance(current_intersection, ndarray) and current_intersection[2]>nearest_intersection[2]):
                nearest_intersection = current_intersection
                normal = plane.normal
        if normal is not None: 
            return nearest_intersection, normal
        else: return False, False

    #get all intersections with a ray
    def get_intersections(self, ray):
        intersections = []
        for pl_index, plane in enumerate(self.planes): 
            current_intersection, current_normal = plane.get_intersection(ray)
            if not self.in_face(current_intersection, plane, pl_index):
                current_intersection = False; current_normal = False
            if isinstance(current_intersection, ndarray):
                intersections.append(current_intersection)
        return intersections

class Light(Shape):
    def __init__(self, point, color='FFFFFF'):
        self.point = point
        self.color = color
        self.rgb = self.get_rgb()

class Camera(object):
    def __init__(self, distance):
        self.point = array([0, 0, distance])

class Screen(object):
    def __init__(self, width, height, res):
        self.width = width
        self.height = height
        self.res = res

    #get position of pixels through which to trace rays
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
        self.sum_lights = self.get_sum_lights()
        self.shapes = shapes

    def get_sum_lights(self):
        '''returns elementwise sum of rgb values for all lights
        capped at 255
        '''
        sum_lights = array([0, 0, 0, 255])
        for light in self.lights: 
            sum_lights = sum_lights + light.rgb
        return minimum(sum_lights, array([255, 255, 255, 255]))

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

    #returns foremost intersection of a ray with a shape in the field
    def get_nearest_intersection(self, point, ray):
        intersections = []

        min_distance = inf
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

    #returns all intersections of a ray with shapes in the field
    def get_intersections(self, ray):
        intersections = []
        for shape in self.shapes:
            intersections += shape.get_intersections(ray)
        return intersections

    #returns true if point is blocked from light by any shape in field
    def isblocked(self, point, light):
        ray = Ray(point, light.point)
        intersections = self.get_intersections(ray)
        for intersection in intersections:
            if get_dist(point, intersection)>.01 and isbetween(intersection, point, light.point):
                return True
        return False

    #returns cosine of angle of incidence between light and shape at point
    def get_cos_theta(self, point, normal, light):
        light_ray = Ray(point, light.point)
        cos_theta = abs(dot(light_ray.direction, normal.direction)
                    /(linalg.norm(light_ray.direction)*linalg.norm(normal.direction)))
        return cos_theta

    #returns direct lighting on shape from light given angle of incidence
    def get_lighting(self, shape, cos_theta, light):
        lighting = minimum(shape.rgb, light.rgb)*cos_theta
        return lighting.astype(int)

    #returns ambient lighting on shape given current illumination
    def get_ambient_light(self, illum, shape):
        light_deficit = self.sum_lights[:3]-illum[:3]
        min_deficit = min(light_deficit)
        benchmark_color = where(light_deficit==min_deficit)
        bump_percent = .2*min_deficit/255
        ambient_light = append(bump_percent*minimum(self.sum_lights[:3], shape.rgb[:3]), 0)
        return ambient_light.astype(int)

    #returns list of RGBA values for pixels in screen
    def get_pixels(self):
        pixels = []

        for point in self.screen.get_points():
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
                illum = illum + self.get_ambient_light(illum, shape)

                pixels.append(tuple(illum))

        return pixels

    #draws image from list of RGBA tuples
    def draw(self, filename):
        mode = 'RGBA'
        size = self.screen.res
        pixels = self.get_pixels()

        my_image = Image.new(mode, size)
        my_image.putdata(pixels)
        my_image.save('test-images/' + filename)

camera = Camera(100)
screen = Screen(100, 100, (500, 500))
screen2 = Screen(100, 100, (1500, 1500))

light0 = Light(array([100, 100, -50]))
light1 = Light(array([100, 100, -60]), color='12ECA9')
light2 = Light(array([-100, 100, -60]), color='FCF76C')
light3 = Light(array([0, 100, -100]))
lights0 = [light0]
lights12 = [light1, light2]
lights3 = [light3]

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

r = Ray(array([1, 1, 1]), array([6, 1, 1]))

normal = Ray(array([0, 0, 0]), array([0, 1, 0]))
plane = Plane(array([0, -75, 0]), normal)

b0 = Block(array([-100, 50, -150]), 50, 20, 30, color='3706AC')
b0.rotate(['x', 'y'], [pi/3, -pi/6])
b1 = Block(array([0, -75, -100]), 50, 30, 50, color='3706AC')
b1.rotate(['y'], [-pi/4])
b2 = Block(array([0, -45, -105]), 40, 25, 40, color='12ECA9')
b2.rotate(['y'], [-pi/4])
b3 = Block(array([0, -20, -120]), 20, 20, 20, color='FE3F3F')
b3.rotate(['y'], [-pi/4])
b4 = Block(array([-500, 0, -20]), 1000, 5, 1500, color='52A437')

shapes0 = [s5, s6, s7, s8, plane]
shapes1 = [b0, plane]
shapes2 = [s5, s6, s7, s8, b0, plane]
shapes3 = [b1, b2, b3, s6, s7, s8, plane]

colored_spheres = World(camera, screen, lights0, shapes0)
colored_spheres_and_light = World(camera, screen, lights12, shapes0)
emmi_world = World(camera, screen, lights3, shapes0)
block_test = World(camera, screen, lights0, shapes1)
blocks_and_spheres = World(camera, screen, lights0, shapes2)
block_stack = World(camera, screen2, lights0, shapes3)
shadow_world = World(camera, screen, lights3, [b4, plane])

block_stack.draw('block_stack.png')

