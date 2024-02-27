import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        
        report = {}

        for i in range(len(list((models)))):
            model = list(models.values())[i] #get each and every model

            model.fit(X_train, y_train) # train model

            y_train_pred = model.predict(X_train) # do the prediction on the train data

            y_test_pred = model.predict(X_test) # do the prediction on the test data

            train_model_score = r2_score(y_train, y_train_pred) # find the r2 for the predictions on the train data

            test_model_score = r2_score(y_test, y_test_pred) # find the r2 for the predictions on the test data

            report[list(models.keys())[i]] = test_model_score # keep appending for all the models

        return report
    
    except Exception as e:
        raise CustomException(e, sys)