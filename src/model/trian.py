from data import download
from features import extract_features

data = download.download_data()

features,labels=extract_features.features_extract(data)

from sklearn.model_selection import train_test_split

def split(features,Lables):
    x_train,x_test,y_train,y_test=train_test_split(features,Lables)
    return x_train,x_test,y_train,y_test


def model():

    pass