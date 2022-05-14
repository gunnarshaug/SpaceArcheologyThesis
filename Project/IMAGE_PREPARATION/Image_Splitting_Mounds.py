import os
import shutil
from pathlib import Path
from PIL import Image


def split_image_mounds(filename,store_path, size, image_number_start, file_type):
    temp_image = Image.open(filename)
    image_width, image_height = temp_image.size
    counter = image_number_start - 1
    for i in range(image_height // size):
        for j in range(image_width // size):
            counter += 1
            box = (j * size, i * size, (j + 1) * size, (i + 1) * size)
            img = Image.new('RGB', (size, size), 255)
            img.paste(temp_image.crop(box))
            path = os.path.join(store_path, str(counter) + '.' + file_type)
            img.save(path)



if __name__ == '__main__':
    split_image_mounds(os.path.join("C:/Bachelor Thesis/Mound PNG/SLRM/temp4.png"),"C:/Bachelor Thesis/Mound PNG/SLRM", 200, 2552, 'png')