import utilities as util
from PIL import Image
import geometry as geometry
import numpy as np
import image_resizer as ir
import cv2
import matplotlib.pyplot as plt

def mosaic(path, percent):
    image_directory = path
    resized_directory = 'REU2023_new_sequence_resized/30m/5_up'
    ir.resize_images_percent(image_directory, resized_directory, percent)
    all_images, image_coords = util.importData(image_directory, resized_directory)
    focal_length, sensor_width, camera_brand, camera_model, date_time, flight_duration = geometry.getMetadata(image_directory)
    print(focal_length, camera_brand, camera_model, date_time, flight_duration)
    altitude = image_coords[0]['altitude']
    cm_per_pixel = (sensor_width * altitude * 100) / (focal_length * all_images[0].width) #GSD Calculation

    print(cm_per_pixel, cm_per_pixel * all_images[0].width / 100, cm_per_pixel * all_images[0].height / 100)
    date, time = date_time.split(' ', 1)
    date = date.replace(':', '-')
    util.get_weather(date, time, image_coords[0]['latitude'], image_coords[0]['longitude'])

    # combined_image = Image.new("RGB", (all_images[0].width * len(all_images), all_images[0].height * len(all_images)))
    # total_vertical = 0
    # total_horizontal = 0

    # sift = cv2.SIFT_create()
    # bf = cv2.BFMatcher()

    # for i in range(len(image_coords) - 1):
    #     dist = geometry.GPStoMeters(image_coords[i]['latitude'], image_coords[i]['longitude'], image_coords[i + 1]['latitude'], image_coords[i + 1]['longitude'])
    #     dist_horizontal = geometry.GPStoMeters(image_coords[i]['latitude'], image_coords[i]['longitude'], image_coords[i]['latitude'], image_coords[i + 1]['longitude'])
    #     dist_vertical = np.sqrt(dist**2 - dist_horizontal**2)
    #     bearing = geometry.GPStoBearing(image_coords[i]['latitude'], image_coords[i]['longitude'], image_coords[i + 1]['latitude'], image_coords[i + 1]['longitude'])
    #     print(dist, dist_horizontal, dist_vertical, bearing)

    #     total_vertical += int(dist_vertical * 100 / cm_per_pixel)
    #     total_horizontal += int(dist_horizontal * 100 / cm_per_pixel)
    #     image = all_images[i].rotate(bearing, expand=True)
    #     combined_image.paste(image, (total_horizontal, total_vertical), image)

    #     kp1, des1 = sift.detectAndCompute(np.array(all_images[i]),None)
    #     kp2, des2 = sift.detectAndCompute(np.array(all_images[i + 1]),None)
    #     matches = bf.knnMatch(des1,des2,k=2)
    #     good = []
    #     for m,n in matches:
    #         if m.distance < 0.75*n.distance:
    #             good.append([m])
    #     img3 = cv2.drawMatchesKnn(np.array(all_images[i]),kp1,np.array(all_images[i + 1]),kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #     plt.imshow(img3),plt.show()

    # combined_image = combined_image.crop(combined_image.getbbox())

    # combined_image.show()
    # combined_image.save('REU2023_new_sequence_resized/30m/5_up/result.png')


if __name__ == "__main__":
    mosaic("REU2023_new_sequence/30m/5_up", 20)