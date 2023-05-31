import csv
import random


class TrackingPointsTask:

    def __init__(self, task_parameters):
        self.n = task_parameters["n"]
        self.task_parameters = task_parameters
        self.locations_storage = []
        self.points_locations = []  # empty list to store random points
        self.labels_list = []
        self.labels_dictionary = {}
        self.initialize_locations_storage_and_labels()

    # Initialize list of nxn locations of points on screen and corresponding labels (0-n^2-1+
    def initialize_locations_storage_and_labels(self):
        n = self.n
        x = self.task_parameters["screen width"] / n
        y = self.task_parameters["screen height"] / n
        label_num = 0
        # create list of points:
        for i in range(0, n):
            for j in range(0, n):
                point = (x/2+x*i, y/2+y*j)
                self.locations_storage.append(point)
                self.labels_dictionary[point] = label_num
                label_num += 1

    # Randomize point from locations_on_screen list
    def randomize_point(self):
        return random.choice(self.locations_storage)

    # Save point location, label and the name of the corresponding image of eye
    def save_point_location(self, filename, random_point):
        self.points_locations.append((filename, random_point[0], random_point[1], self.labels_dictionary[random_point]))
        self.labels_list.append(self.labels_dictionary[random_point])

    def get_labels_list(self):
        return self.labels_list

    # Export an Excel file with images names and points locations
    def export_data_to_excel(self):
        with open('points_locations.csv', 'a', newline='') as file:
            # create a csv writer object
            writer = csv.DictWriter(file, fieldnames=['filename', 'x', 'y', 'label'])
            writer.writeheader()
            for row in self.points_locations:
                writer.writerow({
                    'filename': row[0],
                    'x': row[1],
                    'y': row[2],
                    'label': row[3]
                })

