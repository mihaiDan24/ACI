import matplotlib.pylab as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import scipy.ndimage as ndimage
import cv2 as cv2

print('Reading image...')
 
img_file = ('images.png')
img = plt.imread(img_file)

points = [] 
for i in range(50):
    points.append([np.random.uniform(0, img.shape[0]),np.random.uniform(0, img.shape[1])])
points = np.array(points)

vor = Voronoi(points)

print('Finished drawing voronoi...')
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111)
ax.imshow(ndimage.rotate(img, 180))
voronoi_plot_2d(vor, point_size=10, ax=ax)
plt.show()
