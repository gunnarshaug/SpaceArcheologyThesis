import csv
import os

#This class takes in the result csv of a composite scan of archaeological mounds, and calculates the coordinates in degrees.
#The pixel size is fixed to the pixel size of the standard SAR data generated from Google Earth Engine.
#The top left coordinate of the composite needs to be input.
from pathlib import Path


class Calculate_Burial_Mound_Results:
    def __init__(self, result_file, deg_left, deg_top):
        self.result_file = result_file
        self.pixel_size = 8.98315284119515E-05
        self.deg_left = deg_left
        self.deg_top = deg_top
        self.bbox_list = []
        self.deg_list = []

    def calculate_deg(self):
        for i in range(len(self.bbox_list)):
            left = (self.bbox_list[i][0] * self.pixel_size) + self.deg_left
            right = (self.bbox_list[i][2] * self.pixel_size) + self.deg_left
            top = self.deg_top - (self.bbox_list[i][1] * self.pixel_size)
            bottom = self.deg_top - (self.bbox_list[i][3] * self.pixel_size)
            self.deg_list.append((left, right, top, bottom))


    def read_result_csv(self):
        with open(self.result_file, 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            next(csv_reader)
            for row in csv_reader:
                self.bbox_list.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))

    #This method creates the new result file. All other methods in class are internal to the class.
    def write_new_result_csv(self, filename):
        self.read_result_csv()
        self.calculate_deg()
        with open(filename, 'w', newline='') as write_obj:
            csv_writer = csv.writer(write_obj)
            csv_writer.writerow(['xmin', 'ymin', 'xmax', 'ymax' , 'deg_left', 'deg_top', 'deg_right', 'deg_bottom'])
            for i in range(len(self.bbox_list)):
                csv_writer.writerow([
                    self.bbox_list[i][0],
                    self.bbox_list[i][1],
                    self.bbox_list[i][2],
                    self.bbox_list[i][3],
                    self.deg_list[i][0],
                    self.deg_list[i][2],
                    self.deg_list[i][1],
                    self.deg_list[i][3]
                ])


def calculateCoordinatesForSAR(params):
    source = params.get('csv_path')
    custom_params = params.get('custom_params')
    long, lat = [float(param) for param in custom_params]

    path = Path(source)
    new_file = os.path.join(
        path.parent,
        "results.csv"
    )

    temp = Calculate_Burial_Mound_Results(source, long, lat)
    temp.write_new_result_csv(new_file)
