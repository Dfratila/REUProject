'''
Script that inputs a directory of images and jsons and applies overlap detection methods to determine overlap offsets
After determining overlap offsets, it will evaluate the results against the ground truths in the json file and produce a table

Some assumptions are made:
    -The drone flies at the same altitude for the entirety of the flight
    -The drone is always facing one direction relative to the flight path
    -The name of the flight sequence (e.g. 4_rightup) accurately describes the way the drone is flying relative to where it's facing
        -if the name is 3_up, we will assume that overlap only occurs on the bottom half of the image
'''
import sys
import os
import cv2
import subprocess
import json
import numpy as np
'''
function to read in images from the "label" directory into python objects
returns a list of strings of filenames and a corresponding a list of cv2 images
'''
def importData(directory):
    image_files = []
    cv2_images = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.JPG') or filename.endswith('.jpg'):
            image_files.append(filename)
            image_path = os.path.join(directory, filename)
            cv2_image = cv2.imread(image_path)
            cv2_images.append(cv2_image)
    return image_files, cv2_images

def find_xy(polygon):
    xmin,ymin = 10000,10000
    xmax,ymax = 0,0
    for point in polygon:
        if point[0] < xmin:
            xmin = point[0]
        if point[0] > xmax:
            xmax = point[0]
        if point[1] < ymin:
            ymin = point[1]
        if point[1] > ymax:
            ymax = point[1]
    if xmax == 0 and ymax == 0:
        return []
    return [[xmin,ymin],[xmin,ymax],[xmax,ymax],[xmax,ymin]]
    
'''
Zhiguang's SIFT method
'''
import glob
from tqdm import tqdm
import csv
def sift_zhiguang_overlap(image_dir1,image_dir2, nfeatures=10000, distance = 0.8):
    # Get a list of .jpg files in the directory
    # files = sorted(glob.glob(os.path.join(directory, '*.JPG')))

    # Initialize the SIFT detector
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    resize_radio = 2
    # Load the images in grayscale
    image1 = cv2.imread(image_dir1, 0)
    h,w = image1.shape
    image1 =cv2.resize(image1, (int(w/resize_radio), int(h/resize_radio)))

    image2 = cv2.imread(image_dir2, 0)
    h,w = image2.shape
    image2 = cv2.resize(image2, (int(w/resize_radio), int(h/resize_radio)))

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image1,None)
    kp2, des2 = sift.detectAndCompute(image2,None)

    # Check if keypoints were found in both images
    if des1 is not None and des2 is not None:
        # Initialize the matcher
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)

        # Apply ratio test to find good matches
        good = []
        for m,n in matches:
            if m.distance < distance *n.distance:
                good.append(m)

        # Compute homography
        if len(good)>10:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

            h,w = image1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            a,b,c = dst.shape
            polygon_dup = dst.reshape((a,c))*resize_radio
            polygon_dup = polygon_dup.tolist()
            for i in range(len(polygon_dup)):
                if polygon_dup[i][0] > resize_radio*w:
                    polygon_dup[i][0] = resize_radio*w
                if polygon_dup[i][0] < 0:
                    polygon_dup[i][0] = 0
                if polygon_dup[i][1] > resize_radio*h:
                    polygon_dup[i][1] = resize_radio*h
                if polygon_dup[i][1] < 0:
                    polygon_dup[i][1] = 0
            return find_xy(polygon_dup)

        else:
            print("Not enough matches are found - %d/%d" % (len(good),10))
            return []
    else:
        print("No keypoints were found in one or both images")
        return []
    
    '''
function to run Yang's detection model as a subprocess of current process
'''
def run_command(command):
    process = subprocess.Popen(command, cwd='Bird-Detectron2',shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return process.returncode, stdout.decode('utf-8'), stderr.decode('utf-8')

'''
function to modify bird.json result file to only include entries with a certain amount of confidence (default 0.9)
'''
def parse_json(json_path, confidence):
    with open(json_path, 'r') as json_file:
        file = json.load(json_file)
    filtered_entries = [entry for entry in file if entry.get("confidence", 0.0) >= confidence]
    return filtered_entries

def is_inside_rectangle(point, rectangle):
    x, y = point
    top_left_x, top_left_y = rectangle[0]
    top_right_x, top_right_y = rectangle[1]
    bottom_left_x, bottom_left_y = rectangle[2]
    bottom_right_x, bottom_right_y = rectangle[3]

    return (top_left_x <= x <= top_right_x and
            top_left_y <= y <= bottom_left_y and
            bottom_left_x <= x <= bottom_right_x and
            top_right_y <= y <= bottom_right_y)

def filter_json_list(json_list, rectangle):
    filtered_list = [item for item in json_list if not is_inside_rectangle(item['bbox'][:2], rectangle)]
    return len(filtered_list)
'''
main function
'''
def evaluateImages(directory_path):
    filenames, images = importData(directory_path)
    bird_detector_cmd = f'python3 image_inference_bird.py -p ../{directory_path} -m General_Model -t 0.9'
    run_command(bird_detector_cmd)
    json_path = directory_path + '/General_Model/bird.json'
    bird_json = parse_json(json_path, 0.9)

    info_path = directory_path + '/label/info.txt'
    with open(info_path, 'r') as file:
        lines = file.readlines()
        last_total_birds = int(lines[-1].split('=')[1].strip())
    print(f'Ground truth num birds: {last_total_birds}')
    print(f'No overlap Detection: {len(bird_json)} estimated total birds, {100 * abs((len(bird_json) - last_total_birds) / last_total_birds)}% relative error')

    total_detected = 0
    for i in range(1, len(images)):
        image = os.path.join(directory_path, filenames[i])
        prev_image = os.path.join(directory_path, filenames[i - 1])
        pred_rectangle = sift_zhiguang_overlap(prev_image, image)
        filtered_json = _json = [entry for entry in bird_json if entry.get('image_id') == i + 1]
        total_detected += filter_json_list(filtered_json, pred_rectangle)
    print(f'SIFT overlap Detection: {total_detected} estimated total birds, {100 * abs((total_detected - last_total_birds) / last_total_birds)}% relative error')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the file path to a directory of images and jsons as a command-line argument.")
    else:
        directory_path = sys.argv[1]
        evaluateImages(directory_path)