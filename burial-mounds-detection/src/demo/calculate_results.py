import csv
import os
from pathlib import Path

##COPY##
class CalculateBurialMoundResults:
    """
    Calculates UTM coordinates from a
    :param csv_file:  csv file with columns (xmin, ymin, xmax, ymax)
    :param utm_left: UTM coordinate west 
    :param utm_top: UTM coordinate north
    :param pixel_size: Pixel size of the composite image used with the model to create the csv file.
    """
    def __init__(self, csv_file, utm_left, utm_top, pixel_size):
        self.pixel_size = pixel_size
        self.csv_file = csv_file
        self.utm_left = utm_left
        self.utm_top = utm_top
        self.bbox_list = []
        self.bbox_size_list = []
        self.utm_list = []

    def _calculate_utm(self):
        for i in range(len(self.bbox_list)):
            left = (self.bbox_list[i][0] * self.pixel_size) + self.utm_left
            right = (self.bbox_list[i][2] * self.pixel_size) + self.utm_left
            top = self.utm_top - (self.bbox_list[i][1] * self.pixel_size)
            bottom = self.utm_top - (self.bbox_list[i][3] * self.pixel_size)
            self.utm_list.append((left, right, top, bottom))

    def _calculate_bbox_size(self):
        for i in range(len(self.bbox_list)):
            x_length = (self.bbox_list[i][2] - self.bbox_list[i][0]) * self.pixel_size
            y_length = (self.bbox_list[i][3] - self.bbox_list[i][1]) * self.pixel_size
            if x_length >= y_length:
                max_length = x_length
                ratio = x_length / y_length
            else:
                max_length = y_length
                ratio = y_length / x_length
            self.bbox_size_list.append((max_length, ratio))

    def _read_result_csv(self):
        with open(self.csv_file, 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            next(csv_reader)
            for row in csv_reader:
                self.bbox_list.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))
        self._remove_old_file()

    def write_new_result_csv(self, filename):
        """
        Creates the new result file. All other methods are internal to the class.
        Use the result csv to create a shape file in ArcGIS or QGIS.
        """
        self._read_result_csv()
        self._calculate_bbox_size()
        self._calculate_utm()
        with open(filename, 'w', newline='') as write_obj:
            csv_writer = csv.writer(write_obj)
            csv_writer.writerow(
                ['xmin', 'ymin', 'xmax', 'ymax', 'utm_left', 'utm_right', 'utm_top', 'utm_bottom', 'box_size', 'ratio'])
            for i in range(len(self.bbox_list)):
                csv_writer.writerow([
                    self.bbox_list[i][0],
                    self.bbox_list[i][1],
                    self.bbox_list[i][2],
                    self.bbox_list[i][3],
                    self.utm_list[i][0],
                    self.utm_list[i][1],
                    self.utm_list[i][2],
                    self.utm_list[i][3],
                    self.bbox_size_list[i][0],
                    self.bbox_size_list[i][1],
                ])

    def _remove_old_file(self):
        os.remove(self.csv_file)

def calculateCoordinates(csv_path: str, utm_left: float, utm_top: float, pixel_size: float):

    path = Path(csv_path)
    new_file = os.path.join(
        path.parent,
        "results.csv"
    )

    temp = CalculateBurialMoundResults(csv_path, utm_left, utm_top, pixel_size)
    temp.write_new_result_csv(new_file)
    