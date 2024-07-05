import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def preprocess_data(df):
    remove_features = ['first', 'last', 'street', 'city', 'state', 'zip', 'lat', 'long',
                       'city_pop', 'trans_num', 'unix_time', 'merch_lat',
                       'merch_long', 'trans_date_trans_time', 'gender', 'dob','job',
                       'merchant', 'category']
    df = df.drop(remove_features, axis=1)
    return df

def load_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path, index_col=0)
    df = preprocess_data(df)

    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']

    return X, y
