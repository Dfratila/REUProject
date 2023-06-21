import os
from PIL import Image

def resize_images(input_folder, output_folder, scale_percent):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with Image.open(input_path) as image:
                exif = image.getexif()
                width, height = image.size
                new_width = int(width * scale_percent / 100)
                new_height = int(height * scale_percent / 100)
                new_size = (new_width, new_height)

                image.thumbnail(new_size)
                image.save(output_path, exif=exif)


# input_folder = 'datasets/5_up'
# output_folder = 'datasets_resized/5_up'
# resize_images(input_folder, output_folder, 60)
