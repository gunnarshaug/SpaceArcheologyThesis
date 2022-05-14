import os
import csv

def store_results_to_csv(bboxes, folder):
    headers = ['xmin', 'ymin', 'xmax', 'ymax']
    path = os.path.join(folder, "results.csv")

    with open(path, 'w', encoding='UTF8', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for _, box in enumerate(bboxes):
            xmin, ymin, xmax, ymax, _ = box
            writer.writerow([xmin.item(), ymin.item(), xmax.item(), ymax.item()])

    return path
