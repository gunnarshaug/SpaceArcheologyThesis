import csv

def create_mound_statistics(classification_file, pixel_size):
    with open(classification_file, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        next(csv_reader)
        number_of_mounds = 0
        number_of_round = 0
        number_of_oblong = 0
        size_sum = 0
        less_than_5 = 0
        between_5_and_10 = 0
        between_10_and_15 = 0
        between_15_and_20 = 0
        over_20 = 0
        for row in csv_reader:
            x_range = (int(row[6]) - int(row[4])) * pixel_size
            y_range = (int(row[7]) - int(row[5])) * pixel_size
            if x_range > y_range:
                size = x_range
                size_sum += size
                if (size / y_range) > 1.5:
                    type = "oblong"
                    number_of_oblong += 1
                    number_of_mounds += 1
                else:
                    type = "round"
                    number_of_round += 1
                    number_of_mounds += 1
            else:
                size = y_range
                size_sum += size
                if (size / x_range) > 1.5:
                    type = "oblong"
                    number_of_oblong += 1
                    number_of_mounds += 1
                else:
                    type = "Round"
                    number_of_round += 1
                    number_of_mounds += 1

            if size < 5:
                less_than_5 += 1
            elif size >= 5 and size < 10:
                between_5_and_10 += 1
            elif size >= 10 and size < 15:
                between_10_and_15 += 1
            elif size >= 15 and size < 20:
                between_15_and_20 += 1
            else:
                over_20 += 1

        print(f"Number of mounds: {number_of_mounds}, Number of round mounds: {number_of_round}, Number of oblong mounds: {number_of_oblong}")
        print(f"Number of mounds less than 5m: {less_than_5}")
        print(f"Number of mounds between 5m and 10m: {between_5_and_10}")
        print(f"Number of mounds between 10m and 15m: {between_10_and_15}")
        print(f"Number of mounds between 15m and 20m: {between_15_and_20}")
        print(f"Number of mounds over 20m: {over_20}")
        print(f"Average mound size: {size_sum/number_of_mounds}")






if __name__ == '__main__':
    create_mound_statistics('C:/Bachelor Oppgave/Datasets/Temp1/data/classification.csv',0.25)