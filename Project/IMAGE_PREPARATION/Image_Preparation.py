import os
import shutil
from pathlib import Path
from PIL import Image


def image_splitter(size, path, filetype):
    store_path = Path(os.path.join(path, 'Resized_to_' + str(size)))
    if store_path.exists() and store_path.is_dir():
        shutil.rmtree(store_path)
    os.mkdir(store_path)
    temp_id = path.split('Id-')
    dataset_id = (temp_id[-1].split('-'))[0]
    for file in os.listdir(path):
        if file.endswith('composite.' + filetype):
            filename = os.path.join(path, file)
    split_image(size, filename, store_path, dataset_id, filetype)


def split_image(size, filename, store_path, dataset_id, filetype):
    temp_image = Image.open(filename)
    image_width, image_height = temp_image.size
    for i in range(image_height // size):
        for j in range(image_width // size):
            box = (j * size, i * size, (j + 1) * size, (i + 1) * size)
            img = Image.new('RGB', (size, size), 255)
            img.paste(temp_image.crop(box))
            if j < 10:
                temp_j = '0' + str(j)
            else:
                temp_j = str(j)
            if i < 10:
                temp_i = '0' + str(i)
            else:
                temp_i = str(i)
            path = os.path.join(store_path, 'ID=' + str(dataset_id) + '_x=' + temp_j + '_y=' + temp_i + '.' + filetype)
            img.save(path)


def prepare_images(path, size):
    subfolder_list = [f.path for f in os.scandir(path) if f.is_dir()]
    filetype = 'png'
    for folder in subfolder_list:
        image_splitter(size, folder, filetype)



if __name__ == '__main__':
    prepare_images(os.path.join("c:\\", "Bachelor Oppgave", "Datasets", "Fortresses Iran-Pakistan"),1024)