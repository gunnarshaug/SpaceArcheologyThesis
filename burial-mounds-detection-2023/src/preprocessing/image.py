import cv2, os


class Image:
    def __init__(self, name, path, **kwargs):
        self.path = path
        self.name = name
        self.tile_directory = kwargs["directory"]
        self.tile_type = kwargs["type"]
    
    def save_tiles(self, size):
        images = self._split(size)

        for (count, tile) in enumerate(images):
            tile_name = "{}{}.{}".format(self.name, str(count).zfill(5), self.tile_type)
            tile_path = os.path.join(self.tile_directory, tile_name)
            cv2.imwrite(tile_path, tile)


    def _split(self, size):
        img = cv2.imread(self.path)
        height = img.shape[0]
        width = img.shape[1]
        return [img[height_current:(height_current+size), width_current:(width_current+size)] 
                for height_current in range(0,height-height%size,size)
                for width_current in range(0,width-width%size,size)]

    def _rotate(self, image):
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

if __name__ == "__main__":
    # img = "datasets/slope/test/data/00160.png"
    img = r"C:\Users\egunn\Documents\ArcGIS\Projects\SpaceArcheology\Datasets\Sarpsborg_Halden_2009\Images\Slope_SarpsborgHalden_2009.png"
    tile_opts = {"directory": "datasets/images", "type": "png"}
    image = Image("name", img, **tile_opts)

    image.save_tiles(200)
