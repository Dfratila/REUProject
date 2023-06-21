'''
Driver script. Execute this to perform the mosaic procedure.
'''

import orthomosaic.utilities as util
from PIL import Image
import orthomosaic.geometry as geometry
import numpy as np
import orthomosaic.image_resizer as ir
import cv2


def mosaic(path, percent):
    imageDirectory = path
    resized_directory = path + "/resized"
    ir.resize_images(imageDirectory, resized_directory, percent)
    allImages, imageCoords = util.importData(imageDirectory, resized_directory)
    focalLength, sensorWidth, cameraBrand, cameraModel = geometry.getMetadata(imageDirectory)
    print(focalLength)
    altitude = imageCoords[0]['altitude']
    cm_per_pixel = (sensorWidth * altitude * 100) / (focalLength * allImages[0].width)

    print(cm_per_pixel, cm_per_pixel * allImages[0].width / 100, cm_per_pixel * allImages[0].height / 100)
    print()

    combined_image = Image.new("RGB", (allImages[0].width * len(allImages), allImages[0].height * len(allImages)))
    total_vertical = 0
    total_horizontal = 0

    for i in range(len(imageCoords) - 1):
        dist = geometry.GPStoMeters(imageCoords[i]['latitude'], imageCoords[i]['longitude'], imageCoords[i + 1]['latitude'], imageCoords[i + 1]['longitude'])
        dist_horizontal = geometry.GPStoMeters(imageCoords[i]['latitude'], imageCoords[i]['longitude'], imageCoords[i]['latitude'], imageCoords[i + 1]['longitude'])
        dist_vertical = np.sqrt(dist**2 - dist_horizontal**2)
        bearing = geometry.GPStoBearing(imageCoords[i]['latitude'], imageCoords[i]['longitude'], imageCoords[i + 1]['latitude'], imageCoords[i + 1]['longitude'])
        print(dist, dist_horizontal, dist_vertical, bearing)

        total_vertical += int(dist_vertical * 100 / cm_per_pixel)
        total_horizontal += int(dist_horizontal * 100 / cm_per_pixel)
        image = allImages[i].rotate(bearing, expand=True)
        combined_image.paste(image, (total_horizontal, total_vertical), image)

    combined_image = combined_image.crop(combined_image.getbbox())

    # images = [Image.fromarray(allImages[i]) for i in allImages]
    # size = (len(images) * images[0].size[0], images[0].size[1])

    combined_image.show()
    combined_image.save(path + "result.png")


if __name__ == "__main__":
    mosaic("./datasets/5_up", 20)
