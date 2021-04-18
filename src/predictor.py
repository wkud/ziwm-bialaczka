import numpy
import math
import pandas
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

def predict(data, feature_ranking):
    # alpha = 0.05
    # for index, row in features_with_values.iterrows():
    #     p_value = row['P_values']
    #     if p_value > alpha:
    #         features_with_values.drop(index, inplace=True)

    # print(features_with_values.sort_values('Scores', ascending=False).round(3))

    k_folds = RepeatedStratifiedKFold(n_repeats=5, n_splits=2, random_state=1) # cross validator

    # experiment parameters' variants
    neighbors_counts = [1, 5, 9]
    metrics = ['manhattan', 'euclidean']

    # prepare result table
    column_headers = ['top_feature_count', 'neighbor_count', 'metric', 'scores', 'mean_accuracy', 'mean_confusion_matrix']
    results = pandas.DataFrame(columns=column_headers)


    print('Training models. Please wait...')
    for top_features_count in range(1, feature_ranking.feature_count + 1): # range of integers from 1 (inclusive) to feature_count + 1 (exclusive)
        for neighbor_count in neighbors_counts:
            for metric in metrics:
                knn = KNeighborsClassifier(n_neighbors=neighbor_count, metric=metric)
                current_iteration_scores = []
                current_iteration_confusion_matrices = numpy.zeros(shape=(20, 20))
                number_of_iterations = 0

                # take n best features (with increasing n) in order to find optimal n value (n = top_features_count)
                top_features = feature_ranking.sorted_samples_x_features[:, 0:top_features_count] 

                # perform experiment for given parameters
                for train, test in k_folds.split(top_features, data.class_labels):
                    knn.fit(top_features[train], data.class_labels[train])
                    current_score = knn.score(top_features[test], data.class_labels[test])
                    current_iteration_scores.append(current_score)

                    predicted_class_labels = knn.predict(top_features[test])
                    current_confusion_matrix = confusion_matrix(data.class_labels[test], y_pred=predicted_class_labels)
                    current_iteration_confusion_matrices += current_confusion_matrix

                    number_of_iterations += 1
                
                # save results for this experiment
                results.loc[len(results)] = [top_features_count, 
                                             neighbor_count, 
                                             metric, 
                                             current_iteration_scores,
                                             numpy.array(current_iteration_scores).mean().round(3),
                                             (current_iteration_confusion_matrices / number_of_iterations)]

    # sort results (rows) based on value in 'mean_accuracy' column
    results = results.sort_values('mean_accuracy')
    j = 0
    print('Best mean models scoreboard:')
    for i, row in results.iterrows():
        j += 1
        # 'top_feature_count', 'neighbor_count', 'metric', 'scores', 'mean_accuracy', 'mean_confusion_matrix']
        print(
            f'[{len(results) - j}] Mean score for neighbor_count={row["neighbor_count"]}, metric={row["metric"]}, '
            f'top_feature_count={row["top_feature_count"]}: {row["mean_accuracy"]}')

    return results
