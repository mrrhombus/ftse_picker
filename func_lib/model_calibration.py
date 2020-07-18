import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

class ModelCalibration:
    def __init__(self):
        self.is_calibrated=False
        return

    def predict_with_model(self,x_data):
        if self.is_calibrated==False:
            print("Model needs to be calibrated with ModelCalibration.train_and_predict")
            return
        predictions=self.model.predict(x_data)
        return predictions

    def train_and_predict(self, x_data, y_data, max_leaf_nodes=100):
        self.x_train, self.x_valid, self.y_train, self.y_val = train_test_split(x_data, y_data, random_state=1)
                
        this_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
        this_model.fit(self.x_train, self.y_train)
        test_predictions=this_model.predict(self.x_train)
        train_mae = mean_absolute_error(test_predictions, self.y_train)
        print("Train MAE: {:,.5f}".format(100*train_mae))

        val_predictions = this_model.predict(self.x_valid)
        val_mae = mean_absolute_error(val_predictions, self.y_val)
        print("Validation MAE: {:,.5f}".format(100*val_mae))

        self.model=this_model
        self.train_predictions=test_predictions
        self.validation_predictions=val_predictions
        self.is_calibrated=True
        return this_model