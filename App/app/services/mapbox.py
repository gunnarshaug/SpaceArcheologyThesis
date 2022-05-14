from io import BytesIO
import os
from PIL import Image
import mercantile
from furl import furl
import requests
from app.repositories.mapbox_repository import store_mapbox, get_all, get_by_id, delete_mapbox
from app.repositories.image_repository import get_last_id
from app.services.images import store_image, delete_image
from flask import current_app

zoom = 17
TILE_SIZE = 512

MB_KEY = os.environ['MAPBOX_KEY']
SOURCE_URL = os.environ['MAPBOX_URL']


def get_all_mapboxes():
    return get_all()


def get_mapbox_by_id(id):
    return dict(get_by_id(id))


def store_composite(title, description, coordinates):
    image = download_image(coordinates)
    new_id = get_last_id() + 1
    store_image(image, new_id)
    store_mapbox(title, description, coordinates, new_id)


def delete_composite(id):
    image = get_by_id(id)
    delete_image(image['image_id'])
    delete_mapbox(id)


def download_image(data):
    longitude_top_left, \
    latitude_top_left, \
    longitude_bottom_right, \
    latitude_bottom_right = data

    # Parameters for mercantile.tiles is (east,south,west,north,zoom).
    tile_list = list(
        mercantile.tiles(longitude_top_left, latitude_bottom_right, longitude_bottom_right, latitude_top_left, zoom)
    )

    if len(tile_list) == 0 or len(tile_list) > 500:
        raise ValueError('Number of tiles returned are 0 or larger than 500. Check coordinate input.')

    first_tile, last_tile = tile_list[0], tile_list[-1]
    xMin, yMin = first_tile.x, first_tile.y
    xMax, yMax = last_tile.x, last_tile.y

    rows = yMax - yMin + 1
    cols = xMax - xMin + 1

    composite_image = Image.new('RGB', (TILE_SIZE * cols, TILE_SIZE * rows))

    x_coords = list(dict.fromkeys([tile.x for tile in tile_list]))
    y_coords = list(dict.fromkeys([tile.y for tile in tile_list]))
    for i, tile in enumerate(tile_list):
        tile_img = download_tile(tile)
        x_position = x_coords.index(tile.x) * TILE_SIZE
        y_position = y_coords.index(tile.y) * TILE_SIZE
        composite_image.paste(tile_img, (x_position, y_position))

    return composite_image


def download_tile(tile):
    url = furl(SOURCE_URL) \
        .add(path=str(zoom)) \
        .add(path=str(tile.x)) \
        .add(path="{}@2x.pngraw".format(str(tile.y))) \
        .add(query_params={'access_token': str(MB_KEY)}).url

    response = requests.get(url, stream=True)
    return Image.open(BytesIO(response.content))
