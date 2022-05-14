import csv
import os


class Calculate_Burial_Mound_Results:
    def __init__(self, result_file, pixel_size, utm_left, utm_top):
        self.result_file = result_file
        self.pixel_size = pixel_size
        self.utm_left = utm_left
        self.utm_top = utm_top
        self.file_list = []
        self.bbox_list = []
        self.score_list = []
        self.imsize_list = []
        self.bbox_size_list = []
        self.utm_list = []
        self.confidence_param_list = []
        self.classification = []

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
                ratio = x_length/y_length
            else:
                max_length = y_length
                ratio = y_length/x_length
            self.bbox_size_list.append((max_length, ratio))

    def read_result_csv(self):
        with open(self.result_file, 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            next(csv_reader)
            for row in csv_reader:
                self.file_list.append(row[0])
                self.score_list.append(row[7])
                self.imsize_list.append((row[1], row[2]))
                self.bbox_list.append((float(row[3]), float(row[4]), float(row[5]), float(row[6])))
                self.classification.append(row[8])

    #This method creates the new result file. All other methods in class are internal to the class.
    def write_new_result_csv(self, filename):
        self.read_result_csv()
        self.calculate_bbox_size()
        self.calculate_utm()
        self.calculate_confidence_parameter()
        with open(filename, 'w', newline='') as write_obj:
            csv_writer = csv.writer(write_obj)
            csv_writer.writerow(['filename', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax' , 'score', 'utm_left', 'utm_right', 'utm_top', 'utm_bottom', 'box_size', 'ratio', 'Confidence Number'])
            for i in range(len(self.bbox_list)):
                csv_writer.writerow([
                    self.file_list[i],
                    self.imsize_list[i][0],
                    self.imsize_list[i][1],
                    self.bbox_list[i][0],
                    self.bbox_list[i][1],
                    self.bbox_list[i][2],
                    self.bbox_list[i][3],
                    self.score_list[i],
                    self.utm_list[i][0],
                    self.utm_list[i][1],
                    self.utm_list[i][2],
                    self.utm_list[i][3],
                    self.bbox_size_list[i][0],
                    self.bbox_size_list[i][1],
                    self.confidence_param_list[i],
                    self.classification[i]
                ])

    def calculate_confidence_parameter(self):
        for i in range(len(self.score_list)):
            points = 0
            # points for model scores
            if float(self.score_list[i]) > 0.1 and float(self.score_list[i]) <= 0.3:
                points += 1
            elif float(self.score_list[i]) > 0.3 and float(self.score_list[i]) <= 0.5:
                points += 2
            elif float(self.score_list[i]) > 0.5 and float(self.score_list[i]) <= 0.75:
                points += 2
            elif float(self.score_list[i]) > 0.75:
                points += 6

            #points for bounding box size
            if self.bbox_size_list[i][0] < 4:
                points += 0
            elif self.bbox_size_list[i][0] >= 4 and self.bbox_size_list[i][0] < 6:
                points += 1
            elif self.bbox_size_list[i][0] >= 6 and self.bbox_size_list[i][0] < 50:
                points += 2
            else:
                points += 0

            #points for bound box width vs height ratio. Most burial mounds are round features and hence most bounding boxes should have a ratio close to 1.
            if self.bbox_size_list[i][1] > 2:
                points += 0
            elif self.bbox_size_list[i][1] >= 1.5 and self.bbox_size_list[i][1] < 2:
                points += 1
            else:
                points += 2

            #Should the model score for the bounding box be less than 0.25, the total points will be set to 0.
            if float(self.score_list[i]) < 0.1:
                points = 0

            self.confidence_param_list.append(points)



if __name__ == '__main__':
    input_file = os.path.join('C:/Bachelor Thesis','SpaceArcheology','Datasets','SLRM_RGB_25_March_2022_30_Epochs_Results.csv')
    pixel = 0.25
    new_file = os.path.join('C:/Bachelor Thesis', 'SpaceArcheology', 'Datasets', 'SLRM_test.csv')
    temp = Calculate_Burial_Mound_Results(input_file, pixel, 300000, 6528000)
    temp.write_new_result_csv(new_file)
