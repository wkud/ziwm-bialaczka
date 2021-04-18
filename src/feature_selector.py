import numpy
from sklearn.feature_selection import chi2
from const import csv_file_path, leukaemia_features
import math

class Ranking:
    def __init__(self, feature_ids, chi2_scores, p_values, feature_count):
        self.feature_ids = feature_ids
        self.chi2_scores = chi2_scores
        self.p_values = p_values
        self.feature_count = feature_count

def get_features_ranking(data, print_for_latex=False):
    scores, p_values = chi2(data.samples_x_features, data.class_labels)

    ranking = [(f_id, scores[f_id], p_values[f_id]) for f_id in range(len(scores))] # f_id = feature id (number)
    ranking.sort(key=lambda x: x[1], reverse=True)

    print('Ranking:')
    if not print_for_latex:
        print('Nr\tchi2\tNazwa cechy')

    i = 0
    for feature in ranking:
        (feature_number, score, p_value) = feature
        rounded_score = str(round(score, 3)).replace('.', ',')
        name = leukaemia_features[feature_number]
        i += 1
        if print_for_latex:
            print(f'{i}. & {feature_number+1} & {name} & {rounded_score} \\\\')
        else:
            print(f'{feature_number+1}\t{rounded_score}\t{name}')

    feature_ranking = Ranking(feature_ids=[ranking[i][0] for i in range(len(ranking))],
                              chi2_scores=[ranking[i][1] for i in range(len(ranking))],
                              p_values=[ranking[i][2] for i in range(len(ranking))],
                              feature_count=len(ranking))

    return feature_ranking # object containing sorted lists of features (index from one list correspond to another's list's index)
