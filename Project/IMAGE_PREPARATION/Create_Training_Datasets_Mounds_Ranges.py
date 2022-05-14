import os
import shutil
from pathlib import Path
import csv

class Create_Training_Datasets:
    def __init__(self, target_path, source_path, classification_file_type, image_file_type, test_ranges, val_ranges, train_ranges, total_range):
        self.classification_test_list = []
        self.image_test_list = []
        self.classification_train_list = []
        self.image_train_list = []
        self.classification_val_list = []
        self.image_val_list = []
        self.target_path = Path(target_path)
        self.source_path = Path(source_path)
        self.classification_file_type = classification_file_type
        self.image_file_type = image_file_type
        self.test_ranges = test_ranges
        self.val_ranges = val_ranges
        self.train_ranges = train_ranges
        self.total_range = total_range

    def create_train_and_val_set(self):
        file_dict = {}
        for i in range(self.total_range[0],self.total_range[1]+1):
            file_dict[i] = 0
        for j in range(len(self.test_ranges)):
            min = self.test_ranges[j][0]
            max = self.test_ranges[j][1]
            for k in range(min,max+1):
                file_dict[k] = 1
        for l in range(len(self.val_ranges)):
            min = self.val_ranges[l][0]
            max = self.val_ranges[l][1]
            for m in range(min,max+1):
                file_dict[m] = 2
        for n in range(len(self.train_ranges)):
            min = self.train_ranges[n][0]
            max = self.train_ranges[n][1]
            for o in range(min,max+1):
                file_dict[o] = 3
        if self.target_path.exists() and self.target_path.is_dir():
            shutil.rmtree(self.target_path)
        os.mkdir(self.target_path)
        os.mkdir(os.path.join(self.target_path, 'train'))
        os.mkdir(os.path.join(self.target_path, 'train', 'data'))
        os.mkdir(os.path.join(self.target_path, 'val'))
        os.mkdir(os.path.join(self.target_path, 'val', 'data'))
        os.mkdir(os.path.join(self.target_path, 'test'))
        os.mkdir(os.path.join(self.target_path, 'test', 'data'))
        for file in os.listdir(self.source_path):
            if file.endswith('.' + self.classification_file_type):
                if file_dict.get(int(file.split('.')[0])) == 1:
                    self.classification_test_list.append(os.path.join(self.source_path, file))
                elif file_dict.get(int(file.split('.')[0])) == 2:
                    self.classification_val_list.append(os.path.join(self.source_path, file))
                elif file_dict.get(int(file.split('.')[0])) == 3:
                    self.classification_train_list.append(os.path.join(self.source_path, file))
            elif file.endswith('.' + self.image_file_type):
                if file_dict.get(int(file.split('.')[0])) == 1:
                    self.image_test_list.append(os.path.join(self.source_path, file))
                elif file_dict.get(int(file.split('.')[0])) == 2:
                    self.image_val_list.append(os.path.join(self.source_path, file))
                elif file_dict.get(int(file.split('.')[0])) == 3:
                    self.image_train_list.append(os.path.join(self.source_path, file))

        if self.classification_file_type == 'csv':
            self.combine_csv_classification_files('val')
            self.combine_csv_classification_files('test')
            self.combine_csv_classification_files('train')

        self.move_image_files('val')
        self.move_image_files('test')
        self.move_image_files('train')


    def combine_csv_classification_files(self,dir):
        with open(os.path.join(self.target_path, dir, "classification.csv"), "w", newline='') as write_obj:
            csv_writer = csv.writer(write_obj)
            write_header = True
            if dir == 'test':
                for file in self.classification_test_list:
                    with open(file, 'r') as read_obj:
                        csv_reader = csv.reader(read_obj)
                        if write_header:
                            write_header = False
                        else:
                            next(csv_reader)
                        for row in csv_reader:
                                csv_writer.writerow(row)
            elif dir == 'train':
                for file in self.classification_train_list:
                    with open(file, 'r') as read_obj:
                        csv_reader = csv.reader(read_obj)
                        if write_header:
                            write_header = False
                        else:
                            next(csv_reader)
                        for row in csv_reader:
                                csv_writer.writerow(row)
            else:
                for file in self.classification_val_list:
                    with open(file, 'r') as read_obj:
                        csv_reader = csv.reader(read_obj)
                        if write_header:
                            write_header = False
                        else:
                            next(csv_reader)
                        for row in csv_reader:
                                csv_writer.writerow(row)


    def move_image_files(self,dir):
        if dir == 'test':
            for file in self.image_test_list:
                _, filename = os.path.split(file)
                shutil.copyfile(file, os.path.join(self.target_path, dir, 'data', filename))
        elif dir == 'val':
            for file in self.image_val_list:
                _, filename = os.path.split(file)
                shutil.copyfile(file, os.path.join(self.target_path, dir, 'data', filename))
        else:
            for file in self.image_train_list:
                _, filename = os.path.split(file)
                shutil.copyfile(file, os.path.join(self.target_path, dir, 'data', filename))





if __name__ == '__main__':
    test_ranges = [(1, 16),(159, 172),(2386, 2405)]
    #val_ranges = [(17, 29), (97, 113), (1301, 1345), (2051, 2065), (2181, 2187), (2346, 2352)]
    val_ranges = []
    train_ranges = [(30, 96), (114, 158), (173, 200), (1773, 1810), (2157, 2158), (2159, 2165), (2195, 2208), (2230, 2238), (2266, 2280), (17, 29), (97, 113), (1301, 1345), (2051, 2065), (2181, 2187), (2346, 2352), (2568, 2576)]
    temp = Create_Training_Datasets('C:\Bachelor Thesis\SpaceArcheology\Datasets\Temp1', 'C:\Bachelor Thesis\Mound PNG\SLRM', 'csv', 'png', test_ranges, val_ranges, train_ranges, (1,2567))
    temp.create_train_and_val_set()