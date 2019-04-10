import numpy as np
from skimage import color, filters
import cv2
from skimage import io
from matplotlib import pyplot as plt

cube_sides = ["top.jpg", "left.jpg", "front.jpg", "right.jpg", "back.jpg", "bottom.jpg"]
rotated_imgs = []
for cube_side in cube_sides:
	cube_side = cv2.imread(cube_side)
	RGB_img = cv2.cvtColor(cube_side, cv2.COLOR_BGR2RGB)
	rotated_imgs.append(RGB_img)
imgs = []
plt.imshow(rotated_imgs[0])
plt.show()