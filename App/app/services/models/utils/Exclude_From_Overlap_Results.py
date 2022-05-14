import os
import csv


class Exclude_From_Overlap_Results:
    def __init__(self, file, new_file):
        self.file = file
        self.new_file = new_file


    def read_file(self, file):
        coord_list = []
        with open(file, 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            next(csv_reader)
            for row in csv_reader:
                coord_list.append((float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9])))
        return coord_list

    def compare_coordinates(self, coord_list):
        result_list = []
        for i in range(len(coord_list)):
            result_hits = 0
            x_center = ((coord_list[i][5] - coord_list[i][4]) / 2) + coord_list[i][4]
            y_center = ((coord_list[i][6] - coord_list[i][7]) / 2) + coord_list[i][7]
            for j in range(i+1, len(coord_list)):
                if x_center >= coord_list[j][4] and x_center <= coord_list[j][5] and y_center <= coord_list[j][6] and y_center >= coord_list[j][7]:
                    result_hits +=1
            if result_hits == 0:
                result_list.append((coord_list[i][0], coord_list[i][1], coord_list[i][2], coord_list[i][3], coord_list[i][4], coord_list[i][5], coord_list[i][6], coord_list[i][7], coord_list[i][8], coord_list[i][9]))
        return result_list

    def calculate_results(self):
        coord_list = self.read_file(self.file)
        results = self.compare_coordinates(coord_list)

        self.write_results(results)

    def write_results(self, results):
        with open(os.path.join(self.new_file), 'w', newline='') as write_obj:
            csv_writer = csv.writer(write_obj)
            csv_writer.writerow(['xmin', 'ymin', 'xmax', 'ymax', 'utm_left', 'utm_right', 'utm_top', 'utm_bottom', 'box_size', 'ratio'])
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
                ])



if __name__ == '__main__':
    input_file = os.path.join('C:/', 'SpaceArch New', 'SpaceArcheology', 'Datasets', 'Burial Mound Data Norway', 'Results', 'Slope Additional Rotation Best Model - 120 Epochs', '400 Pixel Scan', 'Composite_Slope_Results_with_Overlap_400_UTM.csv')
    new_file = os.path.join('C:/', 'SpaceArch New', 'SpaceArcheology', 'Datasets', 'Burial Mound Data Norway', 'Results', 'Slope Additional Rotation Best Model - 120 Epochs', '400 Pixel Scan', 'Composite_Slope_Results_without_Overlap_400_UTM_Overlap_Removed.csv')
    test = Exclude_From_Overlap_Results(input_file, new_file)
    test.calculate_results()
