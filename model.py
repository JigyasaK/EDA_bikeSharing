import pickle
from sklearn import model_selection
from bike_eda import Datahandler
from sklearn.linear_model import Lasso, Ridge
import os
import pandas as pd
import argparse
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


#class unitTest

class LinearRegressionModel():
    def __init__(self, save_path):
        self.model = RandomForestRegressor(random_state=0, n_estimators=100)
        self.model_path = save_path

    def get_metrics(self,test_x,test_y):
        res_y = self.model.predict(test_x)
        mse = mean_squared_error(test_y, res_y)
        r2score = r2_score(test_y, res_y)
        print("Mean Squared Error:{:.4f}".format(mse))
        print("R Squared Error:{:.4f}".format(r2score))

    def train(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def predict(self,test_x):
        y_res = self.model.predict(test_x)
        return y_res

    def save(self):
        pickle.dump(self.model, open(self.model_path, 'wb'))

    def load(self):
        pretrained =pickle.load(open(self.model_path, 'rb'))
        return pretrained


def split_data(dataframe, split= True):
    """Given a dataframe, target and predictor variables, and a
    list of attributes to dummify, returns training and test sets.

    Args:
        dataframe: input dataframe

    Returns:
        train and test sets
    """
    # dummify attributes
    dummy_attributes = ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
    dummy_dataframe = dataframe
    for attribute in dummy_attributes:
        dummy_dataframe = pd.concat([dummy_dataframe, pd.get_dummies(dummy_dataframe[attribute], prefix=attribute, drop_first=True)], axis=1)
        dummy_dataframe = dummy_dataframe.drop([attribute], axis =1)

    target = dummy_dataframe['cnt']
    predictor = dummy_dataframe.drop(['cnt'], axis=1)
    if split == True:
        X_train, X_test, y_train, y_test = train_test_split(predictor, target, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test
    else:
        return  predictor, target


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile", help="path to data csv file", type=str)
    parser.add_argument("--mode", help="train or predict", type=str)
    parser.add_argument("--model_path", help= "path to/from where model will be saved/loaded", type=str)
    #parser.add_argument("--metrics_path", help="path to/from where model metrics will \
    #saved/loaded for unit-testing model",type=str,default=None)
    args = parser.parse_args()
    model_save_path = args.model_path
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    datafile = args.datafile
    datahandler = Datahandler(datafile)

    # list of attributes to drop
    drop_attributes = ['atemp','dteday','casual','registered','yr','instant']

    # Clean dataframe
    bikeDF = datahandler.clean(attribute=drop_attributes)

    if args.mode == 'train':
        # get train and test sets
        train_X, test_X, train_y, test_y = split_data(bikeDF)
        # fit model
        model = LinearRegressionModel(model_save_path)
        model.train(train_X, train_y)
        # evaluate model
        model.get_metrics(test_X,test_y)
        #TODO: add unit-test to check if model performance meets some performance benchmark and save the model if it does.
        model.save()

    elif args.mode == 'predict':
        model = LinearRegressionModel(model_save_path)
        pretrained_model = model.load()
        bikeDFdata, bikeDFres = split_data(bikeDF, False)
        res = pretrained_model.predict(bikeDFdata.sample(5))
        print("***************************Predicted hourly count*******************************")
        print(res)
        print("********************************************************************************")