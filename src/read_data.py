import pandas as pandas
import math


def load_data(csv_file, columns):
    data = pandas.read_csv(csv_file, sep=';', usecols=[i for i in range(22)])
    data.columns = columns
    featureID = 1
    for index, row in data.iterrows():
        if math.isnan(row['IDKlasy']):
            data.at[index, 'IDKlasy'] = featureID
        else:
            featureID = row['IDKlasy']
    return (data)
