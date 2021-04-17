import numpy
from sklearn.feature_selection import chi2
from const import csv_file_path, leukaemia_features
import math


def get_features_ranking(data, print_for_latex=False):
    scores, p_values = chi2(data.samples_x_features, data.class_labels)

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
