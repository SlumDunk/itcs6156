from com.app.base import *
import time
CITY_NAME = 'Denver'

if __name__ == '__main__':
    time_start = time.time()
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
    visualize_result(regression_dicts)
    print(f"- time cost: {int((time.time() - time_start) / 3600):.0f}h "
          f"{int(((time.time() - time_start) / 60) % 60):.0f}m "
          f"{int((time.time() - time_start) % 60):.0f}s -")
