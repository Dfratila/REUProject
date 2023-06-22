import orthomosaic.utilities as util
from PIL import Image
import orthomosaic.geometry as geometry
import numpy as np
import orthomosaic.image_resizer as ir


def mosaic(path, percent):
    image_directory = path
    resized_directory = path + "/resized"
    ir.resize_images(image_directory, resized_directory, percent)
    all_images, image_coords = util.importData(image_directory, resized_directory)
    focal_length, sensor_width, camera_brand, camera_model = geometry.getMetadata(image_directory)
    print(focal_length)
    altitude = image_coords[0]['altitude']
    cm_per_pixel = (sensor_width * altitude * 100) / (focal_length * all_images[0].width)

    print(cm_per_pixel, cm_per_pixel * all_images[0].width / 100, cm_per_pixel * all_images[0].height / 100)
    print()

    combined_image = Image.new("RGB", (all_images[0].width * len(all_images), all_images[0].height * len(all_images)))
    total_vertical = 0
    total_horizontal = 0

    for i in range(len(image_coords) - 1):
        dist = geometry.GPStoMeters(image_coords[i]['latitude'], image_coords[i]['longitude'], image_coords[i + 1]['latitude'], image_coords[i + 1]['longitude'])
        dist_horizontal = geometry.GPStoMeters(image_coords[i]['latitude'], image_coords[i]['longitude'], image_coords[i]['latitude'], image_coords[i + 1]['longitude'])
        dist_vertical = np.sqrt(dist**2 - dist_horizontal**2)
        bearing = geometry.GPStoBearing(image_coords[i]['latitude'], image_coords[i]['longitude'], image_coords[i + 1]['latitude'], image_coords[i + 1]['longitude'])
        print(dist, dist_horizontal, dist_vertical, bearing)

        total_vertical += int(dist_vertical * 100 / cm_per_pixel)
        total_horizontal += int(dist_horizontal * 100 / cm_per_pixel)
        image = all_images[i].rotate(bearing, expand=True)
        combined_image.paste(image, (total_horizontal, total_vertical), image)

    combined_image = combined_image.crop(combined_image.getbbox())

    combined_image.show()
    combined_image.save(path + "result.png")


if __name__ == "__main__":
    mosaic("./datasets/5_up", 20)
