import os
from PIL import Image

def resize_images_percent(input_folder, output_folder, scale_percent):
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

def resize_images(directory, target_size):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".JPG"):
                file_path = os.path.join(root, file)
                try:
                    img = Image.open(file_path)
                    img = img.resize(target_size)
                    img.save(file_path)
                except Exception as e:
                    print(f"Error resizing {file_path}: {str(e)}")

resize_images_percent('datasets/60m/11222022','datasets_resized/60m/11222022', 20)
