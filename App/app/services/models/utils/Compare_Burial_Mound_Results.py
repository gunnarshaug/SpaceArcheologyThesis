import os
import csv


class Compare_Results:
    def __init__(self, output_path, file1, file2, file3=None, file4=None):
        self.file1 = file1
        self.file2 = file2
        self.file3 = file3
        self.file4 = file4
        self.output_path = output_path


    def read_file(self, file):
        coord_list = []
        with open(file, 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            next(csv_reader)
            for row in csv_reader:
                coord_list.append((float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9])))
        return coord_list

    def compare_coordinates(self, list1, list2, list3=None, list4=None):
        result_list = []
        for i in range(len(list1)):
            result_hits = 0
            x_center = ((list1[i][5] - list1[i][4]) / 2) + list1[i][4]
            y_center = ((list1[i][6] - list1[i][7]) / 2) + list1[i][7]
            for j in range(len(list2)):
                if x_center >= list2[j][4] and x_center <= list2[j][5] and y_center <= list2[j][6] and y_center >= list2[j][7]:
                    result_hits +=1
            if list3 is not None:
                for k in range(len(list3)):
                    if x_center >= list3[k][4] and x_center <= list3[k][5] and y_center <= list3[k][6] and y_center >= list3[k][7]:
                        result_hits +=1
            if list4 is not None:
                for l in range(len(list4)):
                    if x_center >= list4[l][4] and x_center <= list4[l][5] and y_center <= list4[l][6] and y_center >= list4[l][7]:
                        result_hits +=1
            if result_hits > 0:
                result_list.append((list1[i][0], list1[i][1], list1[i][2], list1[i][3], list1[i][4], list1[i][5], list1[i][6], list1[i][7], list1[i][8], list1[i][9], result_hits))
        return result_list

    def calculate_results(self):
        coord_list1 = self.read_file(self.file1)
        coord_list2 = self.read_file(self.file2)
        if self.file3 is not None:
            coord_list3 = self.read_file(self.file3)
        if self.file4 is not None:
            coord_list4 = self.read_file(self.file4)
        if self.file3 is not None:
            if self.file4 is not None:
                results = self.compare_coordinates(coord_list1, coord_list2, coord_list3, coord_list4)
            else:
                results = self.compare_coordinates(coord_list1, coord_list2, coord_list3)
        else:
            results = self.compare_coordinates(coord_list1, coord_list2)


        self.write_results(results)

    def write_results(self, results):
        with open(os.path.join(self.output_path, 'Model_Common_Burial_Mound_Hits.csv'), 'w', newline='') as write_obj:
            csv_writer = csv.writer(write_obj)
            csv_writer.writerow(['xmin', 'ymin', 'xmax', 'ymax', 'utm_left', 'utm_right', 'utm_top', 'utm_bottom', 'box_size', 'ratio', 'number_of_hits'])
            for i in range(len(results)):
                csv_writer.writerow([
                    results[i][0],
                    results[i][1],
                    results[i][2],
                    results[i][3],
                    results[i][4],
                    results[i][5],
                    results[i][6],
                    results[i][7],
                    results[i][8],
                    results[i][9],
                    results[i][10]
                ])

def compare_burial_mound_results(target_path, src1, src2, src3, src4):

    test = Compare_Results(target_path, src1, src2, src3, src4)
    test.calculate_results()

