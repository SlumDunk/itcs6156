from com.app.base import *
import time

CITY_NAME = 'Denver'

if __name__ == '__main__':
    # load data
    time_start = time.time()
    X, Y = load_data(CITY_NAME)
    X = X.values
    Y = Y.values
    cluster_regression_dict = {}
    for method in cluster_methods:
        # cluster data
        cluster_dict = cluster_data(X, method)
        # split the dataset into different cluster
        dataset_dict = split_data(X, Y, cluster_dict)
        # run regression model for different cluster of data
        regression_dicts = call_regressions(dataset_dict)
        cluster_regression_dict[method] = regression_dicts
    save_result(cluster_regression_dict, CITY_NAME)
    # visualize result
    visualize_result(cluster_regression_dict)
    print(f"- time cost: {int((time.time() - time_start) / 3600):.0f}h "
          f"{int(((time.time() - time_start) / 60) % 60):.0f}m "
          f"{int((time.time() - time_start) % 60):.0f}s -")
