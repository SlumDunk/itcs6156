import numpy as np


class Evaluation(object):
    """

    """

    def __init__(self, mse, rmse, mde, mae, r_score):
        self.mse = mse
        self.rmse = rmse
        self.mde = mde
        self.mae = mae
        self.r_score = r_score

    def __str__(self):
        return 'MSE: ' + self.mse + ', RMSE: ' + self.rmse + ', MDE: ' + self.mde + ', MAE: ' + self.mae + ', R SCORE: ' + self.r_score


class DataCluster(object):
    """

    """

    def __init__(self, x_data, y_data):
        self.x = np.array(x_data)
        self.y = np.array(y_data)


pass
