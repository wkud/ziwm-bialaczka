import numpy
from const import csv_file_path
import math


def __create_class_labels(class_labels):
    last_class = 1
    for i in range(class_labels.shape[0]):
        if math.isnan(class_labels[i]):
            class_labels[i] = last_class
        else:
            last_class = class_labels[i]
    return class_labels


def load_data():
    dataset = numpy.genfromtxt(csv_file_path, delimiter=';')

    # each column represent different feature, each row - different sample
    samples_x_features = dataset[1:, 2:-1]
    # class id column only - samples labeled by class ids
    class_labels = __create_class_labels(dataset[1:, 0])

    return Data(samples_x_features, class_labels)

class Data: # class created for convienient storing data
    def __init__(self, samples_x_features, class_labels): # constructor
        self.samples_x_features = samples_x_features
        self.class_labels = class_labels
        
        self.sample_count = len(class_labels)
