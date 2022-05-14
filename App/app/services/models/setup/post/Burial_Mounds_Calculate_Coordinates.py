import csv
import os
from pathlib import Path

# This class takes in the result csv of a composite scan of burial mounds, and calculates the coordinates in UTM.
# The pixel size of the composite scanned needs to be input.
# The top left coordinate of the composite needs to be input.



class CalculateBurialMoundResults:
    def __init__(self, result_file, utm_left, utm_top, pixel_size):
        self.pixel_size = pixel_size
        self.result_file = result_file
        self.utm_left = utm_left
        self.utm_top = utm_top
        self.bbox_list = []
        self.bbox_size_list = []
        self.utm_list = []

    def calculate_utm(self):
        for i in range(len(self.bbox_list)):
            left = (self.bbox_list[i][0] * self.pixel_size) + self.utm_left
            right = (self.bbox_list[i][2] * self.pixel_size) + self.utm_left
            top = self.utm_top - (self.bbox_list[i][1] * self.pixel_size)
            bottom = self.utm_top - (self.bbox_list[i][3] * self.pixel_size)
            self.utm_list.append((left, right, top, bottom))

    def calculate_bbox_size(self):
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

    def read_result_csv(self):
        with open(self.result_file, 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            next(csv_reader)
            for row in csv_reader:
                self.bbox_list.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))

    # This method creates the new result file. All other methods in class are internal to the class.
    def write_new_result_csv(self, filename):
        self.read_result_csv()
        self.calculate_bbox_size()
        self.calculate_utm()
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


def calculateCoordinatesForBurialMounds(params):
    source = params.get('csv_path')
    custom_params = params.get('custom_params')
    params = [float(param) for param in custom_params]
    long = params[0]
    lat = params[1]
    pixel_size = params[2] if len(params) > 2 else 0.25

    path = Path(source)
    new_file = os.path.join(
        path.parent,
        "results.csv"
    )

    temp = CalculateBurialMoundResults(source, long, lat, pixel_size)
    temp.write_new_result_csv(new_file)
