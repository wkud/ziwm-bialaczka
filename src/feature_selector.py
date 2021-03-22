import numpy
from sklearn.feature_selection import chi2
from const import csv_file_path, leukaemia_features
import math


def __create_class_labels(class_labels):
    last_class = 1
    for i in range(class_labels.shape[0]):
        if math.isnan(class_labels[i]):
            class_labels[i] = last_class
        else:
            last_class = class_labels[i]
    return class_labels


def get_features_ranking(print_for_latex=False):
    dataset = numpy.genfromtxt(csv_file_path, delimiter=';')

    # each column represent different feature, each row - different sample
    samples_x_features = dataset[1:, 2:-1]
    # class id column only - samples labeled by class ids
    class_labels = __create_class_labels(dataset[1:, 0])

    scores, p_values = chi2(samples_x_features, class_labels)

    ranking = [(i, scores[i]) for i in range(len(scores))]
    ranking.sort(key=lambda x: x[1], reverse=True)

    print('Ranking:')
    if not print_for_latex:
        print('Nr\tchi2\tNazwa cechy')

    i = 0
    for feature in ranking:
        (feature_number, score) = feature
        rounded_score = str(round(score, 3)).replace('.', ',')
        name = leukaemia_features[feature_number]
        i += 1
        if print_for_latex:
            print(f'{i}. & {feature_number+1} & {name} & {rounded_score} \\\\')
        else:
            print(f'{feature_number+1}\t{rounded_score}\t{name}')

    return ranking
