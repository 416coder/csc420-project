import cv2
import numpy as np
from matplotlib import pyplot as plt

og_img = cv2.cvtColor(cv2.imread('top.jpg'), cv2.COLOR_BGR2RGB)
img = cv2.imread('top.jpg',0)
edges = cv2.Canny(img[:3200, :2000],300,300)


plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

#plt.show()

indices = np.where(edges != [0])
coordinates = zip(indices[0], indices[1])
# print(list(coordinates))
x_list = list(indices[0])
y_list = list(indices[1])
x_min = min(x_list)
y_min = min(y_list)
x_max = max(x_list)
y_max = max(y_list)
bottom_left = x_min, y_min
bottom_right = x_max, y_min
top_left = x_min, y_max
top_right = x_max, y_max

print(x_max)

plt.figure()
plt.imshow(og_img)
plt.plot((y_min, y_min, y_max, y_max, y_min), (x_min, x_max, x_max, x_min, x_min), 'r')
plt.show()

# coordinates_list = list(coordinates)
# print(coordinates_list)
# min(coordinates_list, key=lambda x_y: x_y[0])
# min_x = min(coordinates_list, key=lambda x_y: x_y[0])[0]

# max_x = min(list(coordinates), key=lambda x_y: x_y[0])
# max_y = min(list(coordinates), key=lambda x_y: x_y[1])[1]
#print(indices)
