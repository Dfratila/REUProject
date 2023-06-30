import os
import orthomosaic.utilities as util
from PIL import Image
import orthomosaic.geometry as geometry
import numpy as np
import orthomosaic.image_resizer as ir
import gpt_scripts.gpt_scripts as gpt
import math
import datetime


def mosaic(path, percent, threshold):
    image_directory = path
    resized_directory = path + "/resized"
    ir.resize_images(image_directory, resized_directory, percent)
    all_images, image_coords = util.importData(image_directory, resized_directory)
    focal_length, sensor_width, camera_brand, camera_model, date = geometry.getMetadata(image_directory)
    print(focal_length)
    altitude = image_coords[0]['altitude']
    cm_per_pixel = (sensor_width * altitude * 100) / (focal_length * all_images[0].width)

    print(cm_per_pixel, cm_per_pixel * all_images[0].width / 100, cm_per_pixel * all_images[0].height / 100)
    print()

    max_length = geometry.GPStoMeters(image_coords[0]['latitude'], image_coords[0]['longitude'], image_coords[len(image_coords) - 1]['latitude'], image_coords[len(image_coords) - 1]['longitude'])
    max_length = max_length * 2 * 100 / cm_per_pixel
    print(max_length)

    combined_image = Image.new("RGB", (int(max_length + all_images[0].width), int(max_length + all_images[0].width)))
    bearing = geometry.GPStoBearing(image_coords[0]['latitude'], image_coords[0]['longitude'], image_coords[1]['latitude'], image_coords[1]['longitude'])
    image = all_images[0].rotate(bearing, expand=True)
    combined_image.paste(image, (int((max_length + all_images[0].width) / 2), int((max_length + all_images[0].width) / 2)), image)
    total_vertical = int((max_length + all_images[0].width) / 2)
    total_horizontal = int((max_length + all_images[0].width) / 2)

    for i in range(1, len(image_coords)):
        dist = geometry.GPStoMeters(image_coords[i - 1]['latitude'], image_coords[i - 1]['longitude'], image_coords[i]['latitude'], image_coords[i]['longitude'])
        dist_horizontal = geometry.GPStoMeters(image_coords[i - 1]['latitude'], image_coords[i - 1]['longitude'], image_coords[i - 1]['latitude'], image_coords[i]['longitude'])
        dist_vertical = np.sqrt(dist**2 - dist_horizontal**2)
        if i < len(image_coords) - 1:
            bearing = geometry.GPStoBearing(image_coords[i]['latitude'], image_coords[i]['longitude'], image_coords[i + 1]['latitude'], image_coords[i + 1]['longitude'])
        else:
            bearing = geometry.GPStoBearing(image_coords[i - 1]['latitude'], image_coords[i - 1]['longitude'], image_coords[i]['latitude'], image_coords[i]['longitude'])
        print(dist, dist_horizontal, dist_vertical, bearing)

        xmult = 1
        ymult = 1

        if bearing <= 90:
            ymult = -1
        elif bearing <= 180:
            xmult = 1
        elif bearing <= 270:
            xmult = -1
        else:
            xmult = -1
            ymult = -1

        total_vertical += int(ymult * dist_vertical * 100 / cm_per_pixel)
        total_horizontal += int(xmult * dist_horizontal * 100 / cm_per_pixel)
        image = all_images[i].rotate(bearing, expand=True)
        combined_image.paste(image, (total_horizontal, total_vertical), image)

    # gpt.analyze(path + "General_Model/bird.json", threshold)

    if not os.path.exists(path + "results/"):
        os.makedirs(path + "results/")

    combined_image = combined_image.crop(combined_image.getbbox())
    combined_image.show()
    combined_image.save(path + "results/result.png")


if __name__ == "__main__":
    mosaic("./orthomosaic/datasets/5_up", 20, 0.9)
