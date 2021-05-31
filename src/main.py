from feature_selector import get_features_ranking
from data_loader import load_data
from predictor import predict
from tstudent import tstudent

# returns data object with 'samples_x_features' and 'class_labels' fields
data = load_data()
ranking = get_features_ranking(data)
prediction_results = predict(data, ranking)
# tstudent(prediction_results)
