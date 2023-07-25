'''
STEP 1- sequence of independent (non-overlapping) images:
    1. Detect waterfowl in the image.
    2. Segment the image into 7 wetland habitats
    3. Map waterfowl to habitats
    4. Calculate/Retrieve flight environmental information including date and time, GPS location, area covered, flight altitude, weather information.
'''
import sys
import subprocess
import os
import json 
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from scipy.spatial import distance
import matplotlib.pyplot as plt
import datetime
import supervision as sv
import time

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

'''
function that takes the mask classifications and bird.json and maps birds to habitats based on the locations of their bounding boxes and the closest mask
returns modified json list and a python dictionary containing the counts of birds in each habitat
modified from step0 to take into account the index of image that we're on
'''
def map_birds(data, masks, idx):
    habitat_counts = {'crop':0, 'open water':0, 'shrub':0,'herbaceous':0,'wooded':0,'other':0, 'harvested crop':0}
    for obj in data:
        if obj['image_id'] == idx:
            x, y, width, height = obj['bbox']
            x += width / 2
            y += height / 2
            distances = []
            habitats = []
            for mask in masks:
                if (mask['predicted'] == None):
                    continue
                trues = np.argwhere(mask['segmentation'])
                distances.append(np.min(distance.cdist([(x, y)], trues, metric='euclidean')))
                habitats.append(mask['predicted'])
            obj['habitat'] = habitats[distances.index(min(distances))]
            habitat_counts[obj['habitat']] += 1
    return data, habitat_counts

'''
function to retrieve relevant metadata from a directory of images
modified from step0 to include flight duration
'''
def getMetadata(directory, filenames):
    image_path = os.path.join(directory, filenames[0])
    with Image.open(image_path) as img:
        if hasattr(img, '_getexif') and img._getexif() is not None:
            exif_data = img._getexif()
            focal_length = exif_data.get(37386)
            sensor_width = 7#13.2               #CURRENTLY NO WAY TO GET SENSOR WIDTH OF CAMERA, may need user input
            make = exif_data.get(271)
            model = exif_data.get(272)
            date_time = exif_data.get(36867)
            img_end = Image.open(os.path.join(directory, filenames[-1]))
            end_date_time = img_end._getexif().get(36867)
            date_format = '%Y:%m:%d %H:%M:%S'
            flight_duration = datetime.strptime(end_date_time, date_format) - datetime.strptime(date_time, date_format)
            return focal_length, sensor_width, make, model, date_time, flight_duration
    return None, None, None, None, None

'''
function to get lat, long, and altitude of starting image and ending image
modified from step 0
'''
def getCoords(directory, filenames):
    from pyexiv2 import Image
    info1 = Image(os.path.join(directory, filenames[0]))
    xmp_info1 = info1.read_xmp()
    info2 = Image(os.path.join(directory, filenames[-1]))
    xmp_info2 = info2.read_xmp()
    return float(xmp_info1['Xmp.drone-dji.GpsLatitude']), float(xmp_info1['Xmp.drone-dji.GpsLongitude']), float(xmp_info1['Xmp.drone-dji.RelativeAltitude'][1:]), float(xmp_info2['Xmp.drone-dji.GpsLatitude']), float(xmp_info2['Xmp.drone-dji.GpsLongitude']), float(xmp_info2['Xmp.drone-dji.RelativeAltitude'][1:])

'''
function to get weather info using OpenWeatherMapAPI
'''
import requests
from datetime import datetime

def get_weather(date_time, latitude, longitude):
    api_key = '5ff992ce7ecdc9c26c036862248db43b'
    unixtime = int(datetime.strptime(date_time, '%Y:%m:%d %H:%M:%S').timestamp())
    url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine?lat={latitude}&lon={longitude}&dt={unixtime}&units=metric&appid={api_key}"
    response = requests.get(url)
    data = response.json()

    temperature = data['data'][0]['temp']
    description = data['data'][0]['weather'][0]['description']
    humidity = data['data'][0]['humidity']
    wind_speed = data['data'][0]['wind_speed']

    date, time = date_time.split(' ', 1)
    print(f"Weather information for {date} at {time} at coords {latitude},{longitude}")
    print(f"Temperature: {temperature} C, {temperature * 9/5 + 32} F")
    print(f"Humidity: {humidity}%")
    print(f"Description: {description}")
    print(f"Wind Speed: {wind_speed} m/s")
    return date, time, temperature, humidity, description, wind_speed

'''
function to resize image to 1/scale_factor so that SAM can generate more accurate masks without running out of memory
second function to resize the generated masks back to original size
'''
def resize_ratio(img, relative_size = 2000):
    max_val = max(img.shape)
    resize_ratio = max_val//relative_size
    return resize_ratio
def resize_img(img, ratio):
    return cv2.resize(img, (int(img.shape[1]/ratio), int(img.shape[0]/ratio)))
def resize_masks(masks, original_image_size):
    original_height, original_width = original_image_size
    resized_masks = []
    for mask in masks:
        resized_mask = {}
        resized_mask['segmentation'] = cv2.resize(mask['segmentation'].astype(float), (original_width, original_height)).astype(bool)
        resized_mask['area'] = np.sum(resized_mask['segmentation'])
        scaling_factor = min(original_image_size[0] / mask['segmentation'].shape[0], original_image_size[1] / mask['segmentation'].shape[1])
        xmin,ymin = mask['bbox'][0], mask['bbox'][1]
        xmax,ymax = xmin + mask['bbox'][2], ymin + mask['bbox'][3]
        new_xmin = int(xmin * scaling_factor)
        new_ymin = int(ymin * scaling_factor)
        new_xmax = int(xmax * scaling_factor)
        new_ymax = int(ymax * scaling_factor)
        resized_mask['bbox'] = [new_xmin, new_ymin, new_xmax - new_xmin, new_ymax - new_ymin]
        resized_masks.append(resized_mask)
    return resized_masks

'''
function to read in images from the directory
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

'''
this function will remove all overlap masks with smaller priority which means this algorithm will keep the most number of filters
'''
def remove_overlaps(masks):
    # Calculate the area of each mask and sort the masks by area
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=False)
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):  # Only compare with subsequent masks
            a = sorted_masks[i]['segmentation']
            b = sorted_masks[j]['segmentation']
            overlap = a & b  # Find the overlap between a and b
            no_overlap = ~overlap  # Find the areas where b does not overlap with a
            sorted_masks[j]['segmentation'] = b & no_overlap  # Remove the overlap from b
    return sorted_masks

'''
functions to split an image into 224x224 regions from top to bottom then left to right
'''
class LargeImageDataset(torch.utils.data.Dataset):
    def __init__(self, img, crop_size, slide_size = 0):
        if slide_size == 0:
            self.slide_size = crop_size
        else:
            self.slide_size = slide_size
        self.dataset = img
        self.crop_size = crop_size
        self.w = self.dataset.shape[1]
        self.h = self.dataset.shape[0]
        self.n_crops_w = (self.w + self.slide_size - 1) // self.slide_size
        self.n_crops_h = (self.h + self.slide_size - 1) // self.slide_size
    def __getitem__(self, index):
        x = (index % self.n_crops_w)
        y = (index // self.n_crops_w)
        rs = x * self.slide_size
        re = rs + self.crop_size
        if re > self.w:
            rs -= re - self.w
        bs = y * self.slide_size
        be = bs + self.crop_size
        if be > self.h:
            bs -= be - self.h
        crop = self.dataset[bs:bs+self.crop_size, rs:rs+self.crop_size, :]
        crop = TF.to_tensor(crop)
        return crop, x, y
    def __len__(self):
        return self.n_crops_w * self.n_crops_h
    def get_img_size(self):
        return self.w, self.h
def merge_model_output_numpy(model, dataloader, device, slide_size, image_size):
    h, w, c = image_size
    outRaster = np.zeros((h, w)).astype(np.uint8)
    model.eval()
    with torch.no_grad():  # Don't track gradients
        for crops, xs, ys in dataloader:
            crops = crops.to(device)  # Send the crops to the device where your model is
            results = model(crops)  # Get the model's output for these crops
            results = torch.argmax(results, axis = 1)
            results = (results.cpu().numpy()).astype(np.uint8)  # Send the results back to the CPU and convert to pixel values
            arr_3d = np.empty((len(results), slide_size, slide_size))
            for i, value in enumerate(results):
                arr_3d[i] = np.full((slide_size, slide_size), value)
            for result, x, y in zip(arr_3d, xs, ys):
                rs = x * slide_size
                re = min((x + 1) * slide_size, w)
                right = re - rs
                bs = y * slide_size
                be = min((y + 1) * slide_size, h)
                bottom = be - bs
                outRaster[bs:be, rs:re] = result[:bottom, :right]             
    return outRaster

'''
main function
'''
def process_image(directory_path):
    start_time = time.time()

    #runs Yang's model
    bird_detector_cmd = f'python3 image_inference_bird.py -p ../{directory_path} -m General_Model -t 0.9'
    run_command(bird_detector_cmd)
    
    #parse the resulting json by first removing entries that don't meet 0.9 confidence score
    json_path = directory_path + '/General_Model/bird.json'
    bird_json = parse_json(json_path, 0.9)
    
    print(f'Bird Detection Execution time: {time.time() - start_time} sec')
    start_time = time.time()

    #load SAM
    torch.cuda.empty_cache()
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"
    sam = sam_model_registry[MODEL_TYPE](checkpoint='segmentation/sam_vit_h_4b8939.pth')
    sam.to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam, points_per_batch=64)

    #load trained habitat classification model and initialize variables for results
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    prediction_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=False)
    prediction_model.classifier = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(output_size=1),
        torch.nn.Flatten(),
        torch.nn.Dropout(p=0.2, inplace=False),
        torch.nn.Linear(in_features=1280, out_features=7),
    )
    prediction_model.load_state_dict(torch.load("segmentation/saved_models/supervised.pth")["state_dict"])
    prediction_model = prediction_model.to(DEVICE)
    categories = ["herbaceous", "open water", "harvested crop", "wooded", "other", "shrub", "crop"]
    total_areas = {'crop':0, 'open water':0, 'shrub':0,'herbaceous':0,'wooded':0,'other':0, 'harvested crop':0}
    colors = {'crop':(255, 255, 0), 'open water':(0,0,255), 'shrub':(255,165,0),'herbaceous':(0,255,0),'wooded':(165,42,42),'other':(128,0,128), 'harvested crop':(255,0,0)}
    reversed_colors = {value: key for key, value in colors.items()}
    colors_named = {6:'yellow', 1:'blue', 5:'orange', 0:'green', 3:'brown', 4:'purple', 2:'red'}
    habitat_counts = {'crop':0, 'open water':0, 'shrub':0,'herbaceous':0,'wooded':0,'other':0, 'harvested crop':0}
    segmented_path = os.path.join(directory_path, 'segmented_images')
    if not os.path.exists(segmented_path):
        os.mkdir(segmented_path)
    classified_path = os.path.join(directory_path, 'classified_images')
    if not os.path.exists(classified_path):
        os.mkdir(classified_path)

    print(f'Model Loading Execution time: {time.time() - start_time} sec')
    start_time = time.time()
    
    #iterate through all the images in directory
    image_paths, images = importData(directory_path)
    for idx,image in enumerate(images):

        #classify entire image by 224x224 crops and highlight the image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        highlighted_img = np.zeros_like(image_rgb)
        crops = LargeImageDataset(image_rgb, 224, 224//2)
        dataloader = torch.utils.data.DataLoader(crops, batch_size=64)
        pixel_classes = merge_model_output_numpy(prediction_model, dataloader, DEVICE, 224//2, image_rgb.shape)
        for l,color in colors_named.items():
            mask = pixel_classes == l
            highlighted_img[mask] = np.array(plt.cm.colors.to_rgba(color))[:3]

        resized_image_rgb = resize_img(image_rgb, resize_ratio(image_rgb))
        masks = mask_generator.generate(resized_image_rgb)
        masks = resize_masks(masks, image_rgb.shape[:2])
        masks = remove_overlaps(masks)

        # generate segmented image visualization to show
        mask_annotator = sv.MaskAnnotator()
        detections = sv.Detections.from_sam(masks) #Detections object with attributes xyxy (np array, bounding boxes of masks), masks (np array, actual masks)
        segmented_image = mask_annotator.annotate(image, detections)

        #iterate through SAM masks, for each mask find the class with the most amount of pixels in that mask and use that to describe the mask
        for mask in masks:
            masked_values = pixel_classes[mask['segmentation']]
            unique_values, value_counts = np.unique(masked_values, return_counts=True)
            most_common_index = np.argmax(value_counts)
            mask['predicted'] = categories[int(unique_values[most_common_index])]
            print(mask['predicted'])
            highlighted_img[mask['segmentation']] = colors[mask['predicted']]
        
        #adds the classified pixels to total_areas dict
        flattened_highlighted_img = highlighted_img.reshape(-1, 3)
        unique_colors, color_counts = np.unique(flattened_highlighted_img, axis=0, return_counts=True)
        for idx, color in enumerate(unique_colors):
            if (color[0], color[1], color[2]) in reversed_colors.keys():
                total_areas[reversed_colors[(color[0], color[1], color[2])]] += color_counts[idx]

        #map the birds and write result to bird.json, print number of birds in each habitat, and show original image vs segmented image vs classified image
        bird_json, habitat_counts_i = map_birds(bird_json, masks, idx + 1)
        for key in habitat_counts:
            habitat_counts[key] += habitat_counts_i[key]
        plt.imsave(os.path.join(segmented_path, image_paths[idx]), segmented_image)
        plt.imsave(os.path.join(classified_path, image_paths[idx]), highlighted_img)

    print(f'Bird Mapping, Segmentation, Classification Execution time: {time.time() - start_time} sec')
    start_time = time.time()

    #extract and show relevant image info
    with open(json_path, 'w') as file:
        json.dump(bird_json, file, indent=4)
    focal_length, sensor_width, camera_make, camera_model, date_time, flight_duration = getMetadata(directory_path, image_paths)
    lat1, long1, altitude1, lat2, long2, altitude2 = getCoords(directory_path, image_paths)
    cm_per_pixel = (sensor_width * altitude1 * 100) / (focal_length * image_rgb.shape[1]) #GSD Calculation
    width_meters, height_meters = cm_per_pixel * image_rgb.shape[1] / 100, cm_per_pixel * image_rgb.shape[0] / 100
    print('Date and Time: ' + date_time)
    print(f'Total flight duration: {flight_duration}')
    print(f'Starting GPS Location: {lat1} degrees latitude, {long1} degrees longitude')
    print(f'Ending GPS Location: {lat2} degrees latitude, {long2} degrees longitude')
    print(f'Flight Altitude: {altitude1} meters relative to ground, flight ended at {altitude2} meters')
    print(f'Area Covered: {width_meters * height_meters} square meters per image, {width_meters * height_meters * len(image_paths)} total square meters')
    for key, value in total_areas.items():
        print(f"{key}: {value / sum(total_areas.values()) * 100} % of total area covered")
    print(f'Bird habitat distribution: {habitat_counts}')
    date, time1, temperature, humidity, description, wind_speed = get_weather(date_time, lat1, long1)

    import openai
    try:
        openai.api_key = "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"
        gpt_prompt = "Given the following data about a drone flight used to detect waterfowls and classify natural habitats, write an informative and comprehensible scientific report about the flight:" \
            f"data: \n Date: {date} \n Start Time: {time1} \n flight duration: {flight_duration} \n starting GPS location: {lat1} degrees latitude, {long1} degrees longitude \n" \
            f"ending GPS location: {lat2} degrees latitude, {long2} degrees longitude \n flight altitude: {altitude1} meters relative to ground \n total area covered: " \
            f"{width_meters * height_meters * len(image_paths)} total square meters, {width_meters * height_meters} square meters per image \n Habitat distribution by percentage: {total_areas} \n" \
            f"Bird habitat distribution by total count: {habitat_counts} \n Weather info: \n Temperature: {temperature} C, {temperature * 9/5 + 32} F \n Humidity: {humidity}% \n" \
            f"Description: {description} \n Wind Speed: {wind_speed} m/s"

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {'role': 'user', 'content': gpt_prompt}
            ],
            temperature=0.0,
            frequency_penalty=0.0,
            max_tokens=2048
        )
    except openai.error.AuthenticationError or FileNotFoundError:
        print("API key missing!")
    else:
        with open('gpt-response3.txt', 'w') as file:
            file.write(response.choices[0].message.content)
    
    print(f'Information Retrieval and Report Generation: {time.time() - start_time} sec')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the file path to a directory of images as a command-line argument.")
    else:
        directory_path = sys.argv[1]
        process_image(directory_path)
