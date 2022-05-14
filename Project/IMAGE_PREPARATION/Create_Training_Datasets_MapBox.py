import os
import shutil
from pathlib import Path
import csv

class Create_Training_Datasets:
    def __init__(self, target_path, source_path, classification_file_type, image_file_type, subfolder_path=None):
        self.classification_list = []
        self.image_list = []
        self.target_path = Path(target_path)
        self.source_path = Path(source_path)
        self.classification_file_type = classification_file_type
        self.image_file_type = image_file_type
        self.subfolder_path = subfolder_path

    def create_train_and_val_set(self):
        if self.target_path.exists() and self.target_path.is_dir():
            shutil.rmtree(self.target_path)
        os.mkdir(self.target_path)
        os.mkdir(os.path.join(self.target_path, 'data'))
        for dir in os.listdir(self.source_path):
            for file in os.listdir(os.path.join(self.source_path, dir, self.subfolder_path)):
                if file.endswith('.' + self.classification_file_type):
                    self.classification_list.append(os.path.join(self.source_path, dir, self.subfolder_path, file))
                elif file.endswith('.' + self.image_file_type):
                    self.image_list.append(os.path.join(self.source_path, dir, self.subfolder_path, file))

        if self.classification_file_type == 'csv':
            self.combine_csv_classification_files()

        self.move_image_files()


    def combine_csv_classification_files(self):
        with open(os.path.join(self.target_path, 'data', "classification.csv"), "w", newline='') as write_obj:
            csv_writer = csv.writer(write_obj)
            write_header = True
            for file in self.classification_list:
                with open(file, 'r') as read_obj:
                    csv_reader = csv.reader(read_obj)
                    if write_header:
                        write_header = False
                    else:
                        next(csv_reader)
                    for row in csv_reader:
                        if row[1] == 'exclude':
                            path, _ = os.path.split(file)
                            self.image_list.remove(os.path.join(path, row[0]))
                        else:
                            csv_writer.writerow(row)


    def move_image_files(self):
        for file in self.image_list:
            _, filename = os.path.split(file)
            shutil.copyfile(file, os.path.join(self.target_path, 'data', filename))






if __name__ == '__main__':
    temp = Create_Training_Datasets('C:\Bachelor Oppgave\Datasets\Temp1', 'C:\Bachelor Oppgave\Datasets\Temp', 'csv', 'png', 'Resized_to_1024')
    temp.create_train_and_val_set()