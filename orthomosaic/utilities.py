import cv2
import os
import numpy as np
from pyexiv2 import Image
from PIL import Image as Im
import re


def importData(imageDirectory, resizedDirectory):
    '''
    :param imageDirectory: Name of the directory where images are stored in string form e.g. "datasets/images/"
    :return:
        allImages: A Python List of NumPy ndArrays containing images.
        imageCoords: A Python List of Python Dictionaries containing GPS coord lookups
    '''
    imageTuples = []
    allImages = []
    imageCoords = []
    focalLength = 0
    sensorWidth = 0
    for filename in os.listdir(imageDirectory):
        f = os.path.join(imageDirectory, filename)
        fResize = os.path.join(resizedDirectory, filename)
        if os.path.isfile(f):
            image = cv2.imread(f)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resize = Im.open(fResize).convert('RGBA')

            imageTuples.append((resize, fResize))

            info = Image(f)
            resizeInfo = Image(fResize)
            exif_info = resizeInfo.read_exif()
            xmp_info = info.read_xmp()
            gps = dict()
            gps['latitude'] = float(xmp_info['Xmp.drone-dji.GpsLatitude'])
            gps['longitude'] = float(xmp_info['Xmp.drone-dji.GpsLongitude'])
            gps['altitude'] = float(xmp_info['Xmp.drone-dji.RelativeAltitude'][1:])
            gps['filename'] = f
            imageCoords.append(gps)

    imageTuples = sorted(imageTuples, key=lambda x: x[1], reverse=True)
    return list(zip(*imageTuples))[0], sorted(imageCoords, key=lambda x: x['filename'])

def display(title, image):
    '''
    OpenCV machinery for showing an image until the user presses a key.
    :param title: Window title in string form
    :param image: ndArray containing image to show
    :return:
    '''

    cv2.namedWindow(title,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title,1920,1080)
    cv2.imshow(title,image)
    cv2.waitKey(400)
    cv2.destroyWindow(title)

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    Makes an image with matched features denoted.
    drawMatches() is missing in OpenCV 2. This boilerplate implementation taken from http://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for m in matches:

        # Get the matching keypoints for each of the images
        img1_idx = m.queryIdx
        img2_idx = m.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        radius = 8
        thickness = 3
        color = (255,0,0) #blue
        cv2.circle(out, (int(x1),int(y1)), radius, color, thickness)
        cv2.circle(out, (int(x2)+cols1,int(y2)), radius, color, thickness)

        # Draw a line in between the two points
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), color, thickness)

    # Also return the image if you'd like a copy
    return out