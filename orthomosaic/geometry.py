import datetime

import numpy as np
import cv2
import math as m
from PIL import Image
import os
import sys
path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)
from gpt_scripts import gpt_scripts


def computeUnRotMatrix(pose):
    '''
    See http://planning.cs.uiuc.edu/node102.html. Undoes the rotation of the craft relative to the world frame.
    :param pose: A 1x6 NumPy ndArray containing pose information in [X,Y,Z,Y,P,R] format
    :return: A 3x3 rotation matrix that removes perspective distortion from the image to which it is applied.
    '''
    a = pose[3] * np.pi / 180  # alpha
    b = pose[4] * np.pi / 180  # beta
    g = pose[5] * np.pi / 180  # gamma
    # Compute R matrix according to source.
    Rz = np.array(([m.cos(a), -1 * m.sin(a), 0],
                   [m.sin(a), m.cos(a), 0],
                   [0, 0, 1]))

    Ry = np.array(([m.cos(b), 0, m.sin(b)],
                   [0, 1, 0],
                   [-1 * m.sin(b), 0, m.cos(b)]))

    Rx = np.array(([1, 0, 0],
                   [0, m.cos(g), -1 * m.sin(g)],
                   [0, m.sin(g), m.cos(g)]))
    Ryx = np.dot(Rx, Ry)
    R = np.dot(Rz, Ryx)  # Care to perform rotations in roll, pitch, yaw order.
    R[0, 2] = 0
    R[1, 2] = 0
    R[2, 2] = 1
    Rtrans = R.transpose()
    InvR = np.linalg.inv(Rtrans)
    # Return inverse of R matrix so that when applied, the transformation undoes R.
    return InvR


def warpPerspectiveWithPadding(image, transformation):
    '''
    When we warp an image, its corners may be outside of the bounds of the original image. This function creates a new image that ensures this won't happen.
    :param image: ndArray image
    :param transformation: 3x3 ndArray representing perspective trransformation
    :param kp: keypoints associated with image
    :return: transformed image
    '''

    height = image.shape[0]
    width = image.shape[1]
    corners = np.float32([[0, 0], [0, height], [width, height], [width, 0]]).reshape(-1, 1,
                                                                                     2)  # original corner locations

    warpedCorners = cv2.perspectiveTransform(corners, transformation)  # warped corner locations
    [xMin, yMin] = np.int32(warpedCorners.min(axis=0).ravel() - 0.5)  # new dimensions
    [xMax, yMax] = np.int32(warpedCorners.max(axis=0).ravel() + 0.5)
    translation = np.array(
        ([1, 0, -1 * xMin], [0, 1, -1 * yMin], [0, 0, 1]))  # must translate image so that all of it is visible
    fullTransformation = np.dot(translation, transformation)  # compose warp and translation in correct order
    result = cv2.warpPerspective(image, fullTransformation, (xMax - xMin, yMax - yMin))
    return result


def getMetadata(image_path):
    sorted_dir = sorted(os.listdir(image_path))
    with Image.open(os.path.join(image_path, sorted_dir[0])) as img:
        if hasattr(img, '_getexif') and img._getexif() is not None:
            exif_data = img._getexif()
            focal_length = exif_data.get(37386)
            sensor_width = 7#13.2
            make = exif_data.get(271)
            model = exif_data.get(272)
            date_time = exif_data.get(36867)
            img_end = Image.open(os.path.join(image_path, sorted_dir[-1]))
            end_date_time = img_end._getexif().get(36867)
            date_format = '%Y:%m:%d %H:%M:%S'
            flight_duration = datetime.strptime(end_date_time, date_format) - datetime.strptime(date_time, date_format)
            return focal_length, sensor_width, make, model, date_time, flight_duration
    return None, None, None, None, None


def GPStoMeters(lat1, lon1, lat2, lon2):
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return (c * r) * 1000


def GPStoBearing(lat1, lon1, lat2, lon2):
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    bearing = np.arctan2(np.sin(lon2 - lon1) * np.cos(lat2),
                         np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1))
    bearing = np.degrees(bearing)
    bearing = (bearing + 360) % 360
    return bearing