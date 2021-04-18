from feature_selector import get_features_ranking
from data_loader import load_data
from predictor import predict

data = load_data() # returns data object with 'samples_x_features' and 'class_labels' fields
ranking = get_features_ranking(data)
predict(data, ranking)


