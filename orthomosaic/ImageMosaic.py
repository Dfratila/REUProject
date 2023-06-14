'''
Driver script. Execute this to perform the mosaic procedure.
'''

import utilities as util
from PIL import Image
import geometry
import numpy as np
import cv2

imageDirectory = "./datasets/15m/12312021/"
allImages, imageCoords = util.importData(imageDirectory)
focalLength, sensorWidth, cameraBrand, cameraModel = geometry.getMetadata(imageDirectory)
altitude = 15
cm_per_pixel = (sensorWidth * altitude * 100) / (focalLength * allImages[0].shape[1])
print(cm_per_pixel, cm_per_pixel * allImages[0].shape[0] / 100, cm_per_pixel * allImages[0].shape[1] / 100)
for i in range(len(imageCoords) - 1):
    dist = geometry.GPStoMeters(imageCoords[i]['latitude'], imageCoords[i]['longitude'], imageCoords[i + 1]['latitude'], imageCoords[i + 1]['longitude'])
    dist_horizontal = geometry.GPStoMeters(imageCoords[i]['latitude'], imageCoords[i]['longitude'], imageCoords[i]['latitude'], imageCoords[i + 1]['longitude'])
    dist_vertical = np.sqrt(dist**2 - dist_horizontal**2)
    bearing = geometry.GPStoBearing(imageCoords[i]['latitude'], imageCoords[i]['longitude'], imageCoords[i + 1]['latitude'], imageCoords[i + 1]['longitude'])
    print(dist,dist_horizontal,dist_vertical, bearing)