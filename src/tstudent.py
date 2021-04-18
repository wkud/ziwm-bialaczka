import math
import statistics
from pandas import pandas
import numpy as numpy
from scipy import stats as stats
from scipy.stats import ttest_ind
from tabulate import tabulate


class BestResults:
    def __init__(self, k1euk, k5euk, k9euk, k1man, k5man, k9man):
        self.results = [
            k1euk,
            k5euk,
            k9euk,
            k1man,
            k5man,
            k9man
        ]
        self.column_names = [
            "K = 1, Euklidesowa",
            "K = 5, Euklidesowa",
            "K = 9, Euklidesowa",
            "K = 1, Manhattan",
            "K = 5, Manhattan",
            "K = 9, Manhattan",
        ]
        self.t_statistics = numpy.zeros(
            (len(self.column_names), len(self.column_names)))
        self.p_values = numpy.zeros(
            (len(self.column_names), len(self.column_names)))
        self.column_keys = numpy.array([[key] for key in self.column_names])

    def print_statistics(self):
        for i in range(len(self.column_names)):
            for j in range(len(self.column_names)):
                self.t_statistics[i, j], self.p_values[i, j] = ttest_ind(self.results[i]['scores'],
                                                                         self.results[j]['scores'])
        print("\n\n--------T-statistics--------\n")
        t_statistic_table = numpy.concatenate(
            (self.column_keys, self.t_statistics), axis=1)
        t_statistic_table = tabulate(
            t_statistic_table, self.column_names, floatfmt=".2f", tablefmt=format)
        print(t_statistic_table, "\n")

        print("\n\n--------P-values--------\n")
        p_values_table = numpy.concatenate(
            (self.column_keys, self.p_values), axis=1)
        p_values_table = tabulate(
            p_values_table, self.column_names, floatfmt=".2f", tablefmt=format)
        print(p_values_table, "\n")

        print("\n\n--------Advantages--------\n")
        advantages = numpy.zeros(
            (len(self.column_names), len(self.column_names)))
        for i in range(len(self.column_names)):
            for j in range(len(self.column_names)):
                if self.t_statistics[i, j] > 0:
                    advantages[i, j] = 1
        advantages_table = tabulate(numpy.concatenate(
            (self.column_keys, advantages), axis=1), self.column_names, tablefmt=format)
        print(advantages_table, "\n")

        print("\n\n--------Significance--------\n")
        significance = numpy.zeros(
            (len(self.column_names), len(self.column_names)))
        for i in range(len(self.column_names)):
            for j in range(len(self.column_names)):
                if self.p_values[i, j] <= .05:
                    significance[i, j] = 1
        significance_table = tabulate(numpy.concatenate(
            (self.column_keys, significance), axis=1), self.column_names, tablefmt=format)
        print(significance_table, "\n")

        print("\n\n--------Statistically better--------\n")
        stat_better = numpy.zeros(
            (len(self.column_names), len(self.column_names)))
        for i in range(len(self.column_names)):
            for j in range(len(self.column_names)):
                stat_better[i, j] = significance[i, j] * advantages[i, j]
        stat_better_table = tabulate(numpy.concatenate(
            (self.column_keys, stat_better), axis=1), self.column_names, tablefmt=format)
        print(stat_better_table, "\n")


def find_best_statistically_significant_model(df):
    alfa = 0.05
    df = df.sort_values('mean_accuracy', ascending=False)
    for i, row1 in df.iterrows():
        for j, row2 in df.iterrows():
            if i != j and i < j:
                t, p_value = ttest_ind(row1['scores'], row2['scores'])
                if p_value < alfa:
                    print(
                        f'Comparing [k={row1["neighbor_count"]}, m={row1["metric"]}, f={row1["top_feature_count"]},'
                        f' accuracy={row1["mean_accuracy"]}] with [k={row2["neighbor_count"]}, m={row2["metric"]},'
                        f' f={row2["top_feature_count"]}, accuracy={row2["mean_accuracy"]}]')
                    print(
                        f'\tt-statistic: {t}, p-value: {p_value}, alfa: {alfa}')
                    return


def find_best_model_for_clasificator(df):
    df = df.sort_values('mean_accuracy', ascending=False)
    k1euk = df[(df['neighbor_count'] == 1) & (
        df['metric'] == 'euclidean')].iloc[0]
    k5euk = df[(df['neighbor_count'] == 5) & (
        df['metric'] == 'euclidean')].iloc[0]
    k9euk = df[(df['neighbor_count'] == 9) & (
        df['metric'] == 'euclidean')].iloc[0]
    k1man = df[(df['neighbor_count'] == 1) & (
        df['metric'] == 'manhattan')].iloc[0]
    k5man = df[(df['neighbor_count'] == 5) & (
        df['metric'] == 'manhattan')].iloc[0]
    k9man = df[(df['neighbor_count'] == 9) & (
        df['metric'] == 'manhattan')].iloc[0]
    return BestResults(k1euk, k5euk, k9euk, k1man, k5man, k9man)


def compare_every_model_paired(df, n):
    alfa = 0.05
    statistical_significant_pairs = 0
    statistical_insignificant_pairs = 0
    i_number = 0
    j_number = 0
    df = df.sort_values('mean_accuracy', ascending=False)
    for i, row1 in df.iterrows():
        if i_number > n:
            break
        i_number = i_number + 1
        j_number = 0
        for j, row2 in df.iterrows():
            if j_number > n:
                break
            j_number = j_number + 1
            if i != j and i < j:
                t, p_value = ttest_ind(row1['scores'], row2['scores'])
                if p_value < alfa:
                    statistical_significant_pairs = statistical_significant_pairs + 1
                    print(
                        f'Comparing [k={row1["neighbor_count"]}, m={row1["metric"]}, f={row1["top_feature_count"]}] with'
                        f' [k={row2["neighbor_count"]}, m={row2["metric"]}, f={row2["top_feature_count"]}]')
                    print(
                        f'\tt-statistic: {t}, p-value: {p_value}, alfa: {alfa} -> {p_value} (p) < {alfa} (a)')
                else:
                    statistical_insignificant_pairs = statistical_insignificant_pairs + 1

    print(
        f'Statistical significant pairs = {statistical_significant_pairs}, '
        f'statistical insignificant pairs = {statistical_insignificant_pairs}')


def tstudent(predictions):
    print('Comparing models...\n\n')
    compare_every_model_paired(predictions, 7)
    print('Finding statistics...\n\n')
    find_best_statistically_significant_model(predictions)

    best_model_params = predictions.sort_values(
        'mean_accuracy', ascending=False).iloc[0]
    print(f'\nBest score: {best_model_params["mean_accuracy"]}')
    print(f'Best parameters: metric - {best_model_params["metric"]}, neighbor_count - {best_model_params["neighbor_count"]}, '
          f'amount of features - {best_model_params["top_feature_count"]}')
    best_result = find_best_model_for_clasificator(predictions)
    best_result.print_statistics()
