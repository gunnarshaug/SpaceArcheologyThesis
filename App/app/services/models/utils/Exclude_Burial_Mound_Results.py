import os
import csv
from pathlib import Path


class Exclude_Results:
    def __init__(self, file, output_file, min_size, max_size, ratio_cutoff):
        self.file = file
        self.output_file = output_file
        self.min_size = min_size
        self.max_size = max_size
        self.ratio_cutoff = ratio_cutoff


    def read_file(self, file):
        coord_list = []
        with open(file, 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            next(csv_reader)
            for row in csv_reader:
                coord_list.append((float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9])))
        return coord_list

    def calculate_cutoff(self, record):
        if record[8] < self.min_size or record[8] > self.max_size or record[9] > self.ratio_cutoff:
            return True
        else:
            return False

    def exclude_results(self):
        data = self.read_file(self.file)
        new_data = []
        for i in range(len(data)):
            result = self.calculate_cutoff(data[i])
            if result is False:
                new_data.append(data[i])

        self.write_results(new_data)

    def write_results(self, results):
        with open(os.path.join(self.output_file), 'w', newline='') as write_obj:
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

def exclude_burial_mounds_by_filters(source, target_path, min_size, max_size, ratio_cutoff, id):
    path = Path(source)

    new_file = os.path.join(
        target_path,
        "filtered_{}_{}".format(id, path.name)
    )
    test = Exclude_Results(source, new_file, min_size, max_size, ratio_cutoff)
    test.exclude_results()
    return new_file

