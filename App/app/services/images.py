import os
import cv2
import numpy as np

from app.repositories.image_repository import store_image as store_image_to_db, delete_image as delete_db_image, \
    get_by_id


IMAGES_PATH = os.environ.get('IMAGES_PATH')
TEMP_PATH = os.environ.get('TEMP_PATH')
ANALYZED_IMAGES_PATH = os.environ.get('ANALYZED_PATH')


def get_image_by_name(image_name):
    image_path = os.path.join(IMAGES_PATH, image_name)


def load_image_by_name(image_name):
    image_path = os.path.join(IMAGES_PATH, image_name)
    return cv2.imread(image_path)


def store_image(image, name):
    image_name = "{}.png".format(name)
    image_path = os.path.join(IMAGES_PATH, image_name)
    image.save(image_path)
    store_image_to_db(image_name)


def delete_image(id):
    image = get_by_id(id)
    id, image_path = image['id'], image['path']
    image_path = os.path.join(IMAGES_PATH, image_path)
    os.remove(image_path)
    delete_db_image(id)


def start_points(size, split_size, overlap):
    points = [0]
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
            counter += 1
    return points


def split_image(image, overlap, cut_size, with_storage=False):
    temp_dir = os.path.join(
        TEMP_PATH,
        'parts'
    )

    img_height, img_width, _ = image.shape
    x_points = start_points(img_width, cut_size, overlap)
    y_points = start_points(img_height, cut_size, overlap)
    images = [[[] for i in range(len(x_points))] for i in range(len(y_points))]

    if with_storage:
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))

    for i, y_point in enumerate(y_points):
        for j, x_point in enumerate(x_points):
            split = image[y_point:y_point + cut_size, x_point:x_point + cut_size]
            images[i][j] = split

            if with_storage:
                image_path = os.path.join(
                    temp_dir,
                    '{}_{}.png'.format(i, j)
                )
                cv2.imwrite(image_path, split)

    return np.array(images)


def save_image_with_bboxes(image, bboxes, folder):
    for box in bboxes:
        x_min, y_min, x_max, y_max, label = box
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

    image_path = os.path.join(folder, "results.png")
    cv2.imwrite(image_path, image)
    return image_path
