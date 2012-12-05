from __future__ import division
from numpy import ndarray, array, linalg, dot, arange, pi, sin, cos, tan, sqrt, inf, column_stack, newaxis
from PIL import Image

def get_lighting(shape_color, cos_theta, light_color):
		red = int(min(int(shape_color[:2], 16), int(light_color[:2], 16))*cos_theta)
		green = int(min(int(shape_color[2:4], 16), int(light_color[2:4], 16))*cos_theta)
		blue = int(min(int(shape_color[4:], 16), int(light_color[4:], 16))*cos_theta)
		return array([red, green, blue, 255])

def get_pixels(shape_color, light_color):
	pixels = []
	for row in range(500):
		print row,
		cos_theta = cos((pi*row)/(2*500))
		print cos_theta
		for column in range(500):
			pixel = get_lighting(shape_color, cos_theta, light_color)
			pixels.append(tuple(pixel))
	return pixels

def draw():
		mode = 'RGBA'
		size = (500, 500)
		pixels = get_pixels('3706AC', 'FFFFFF')

		my_image = Image.new(mode, size)
		my_image.putdata(pixels)
		my_image.save('colortest.png')

draw()