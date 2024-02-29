import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:

        report = {}

        for i in range(len(list((models)))):
            model = list(models.values())[i] #get each and every model
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train) # train model

            y_train_pred = model.predict(X_train) # do the prediction on the train data

            y_test_pred = model.predict(X_test) # do the prediction on the test data

            train_model_score = r2_score(y_train, y_train_pred) # find the r2 for the predictions on the train data

            test_model_score = r2_score(y_test, y_test_pred) # find the r2 for the predictions on the test data

            report[list(models.keys())[i]] = test_model_score # keep appending for all the models

        return report
    
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path): # just opening that file path in read mode and it is loading the pickle file by using dill
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)