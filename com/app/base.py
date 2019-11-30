import pandas as pd

from com.models.models import DataCluster
from itcs6156 import settings
from com.machinelearning.clustermodel.cluster_functions import *
from com.machinelearning.regressionmodel.functions import *

cluster_methods = ['kmeans', 'dbscan', 'gmm', 'hierachical']

regression_methods = ['linear', 'decision_tree', 'support_vector', 'gradient_boosting', 'random_forest', 'ridge', 'cnn',
                      'rnn']


def load_data(city_name):
    """
    read data from pkl file
    :return:
    """
    X = pd.read_pickle(settings.DATA_URL + city_name + '/output/train_x.pkl')
    Y = pd.read_pickle(settings.DATA_URL + city_name + '/output/train_y.pkl')
    return X, Y


def cluster_data(X, method):
    """
    call cluster function
    :param X:
    :return:
    """
    cluster_dict = {}

    if method == 'kmeans':
        y_pred = k_means_cluster(X)
    elif method == 'dbscan':
        y_pred = dbscan_cluster(X)
    elif method == 'gmm':
        y_pred = gmm_cluster(X)
    else:
        y_pred = hierarchical_cluster(X)

    # label the data
    labels = np.unique(y_pred)
    for row, label in enumerate(y_pred):
        if label in cluster_dict.keys():
            values = cluster_dict.get(label)
            values.append(row)
            cluster_dict[label] = values
        else:
            values = list()
            values.append(row)
            cluster_dict[label] = values

    return cluster_dict


def regression_data(X, Y, method):
    """
    apply regression model to the dataset and return the result
    :param X:
    :param Y:
    :param method:
    :return:
    """
    Y = Y[:, 1]
    if method == 'linear':
        evaluation_res = linear_regression(X, Y)
    elif method == 'decision_tree':
        evaluation_res = decision_tree_regression(X, Y, 5)
    elif method == 'support_vector':
        evaluation_res = support_vector_regression(X, Y)
    elif method == 'gradient_boosting':
        evaluation_res = gradient_boosting_regression(X, Y)
    elif method == 'random_forest':
        evaluation_res = random_forest_regression(X, Y, 5)
    elif method == 'ridge':
        evaluation_res = ridge_regression(X, Y)
    elif method == 'cnn':
        evaluation_res = cnn_regression(X, Y)
    else:
        evaluation_res = rnn_regression(X, Y)
    return evaluation_res


def visualize_result(regression_result_dict):
    return


def split_data(X, Y, cluster_dict):
    """

    :param X:
    :param Y:
    :param cluster_dict:
    :return:
    """
    dataset_dict = {}
    for key, values in cluster_dict.items():
        x_data = list()
        y_data = list()
        for row in values:
            x_data.append(X[row])
            y_data.append(Y[row])
        dataset_dict[key] = DataCluster(x_data, y_data)
    return dataset_dict


pass


def call_regressions(dataset_dict):
    """
    call regression methods for different clusters of data
    :return:
    """
    regression_dict = {}
    sorted(dataset_dict.keys())
    for label, dataset in dataset_dict.items():
        print('-------------------------cluster: ' + str(label) + '------------------------')
        result_dict = {}
        for method in regression_methods:
            print('-------------------------method: ' + method + '------------------------')
            result_dict[method] = regression_data(dataset.x, dataset.y, method)
        regression_dict[label] = result_dict
        # regression_data(dataset.x, dataset.y, 'rnn')
    return regression_dict


pass
