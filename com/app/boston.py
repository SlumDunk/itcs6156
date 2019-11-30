from com.app.base import *

CITY_NAME = 'Boston'

if __name__ == '__main__':
    # load data
    X, Y = load_data(CITY_NAME)
    X = X.values
    Y = Y.values
    # cluster data
    cluster_dict = cluster_data(X, 'kmeans')
    # split the dataset into different cluster
    dataset_dict = split_data(X, Y, cluster_dict)
    # run regression model for different cluster of data
    regression_dicts = call_regressions(dataset_dict)
    # visualize result
    visualize_result()
