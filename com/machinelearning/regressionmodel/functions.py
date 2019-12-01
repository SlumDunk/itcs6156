import keras
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn import metrics
import numpy as np
from keras.layers import Dense, Activation, Flatten, Conv1D, MaxPooling1D, Dropout, LSTM, TimeDistributed
from keras.models import Sequential
from keras.optimizers import Adam

# super parameters
from com.models.models import Evaluation
from keras import losses

BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 20
LR = 0.006

max_iteration = 10


def linear_regression(x, y):
    """
     function to apply linear regression model to predict the targets
    :param x:
    :param y:
    :return:
    """
    lin_reg = LinearRegression()

    sum_criterion = 0
    sum_test_rmse = 0

    for i in range(max_iteration):
        x_test, x_train, x_tv, x_validation, y_test, y_train, y_tv, y_validation = dataset_split(x, y)
        lin_reg.fit(x_train, y_train)
        y_pred_train = evaluate(lin_reg.predict(x_train), y_train).rmse
        y_pred_validation = evaluate(lin_reg.predict(x_validation), y_validation).rmse
        train_criterion = weighted_cost(y_pred_train, y_pred_validation)

        lin_reg.fit(x_tv, y_tv)
        y_pred_test = lin_reg.predict(x_test)

        sum_criterion = sum_criterion + train_criterion
        sum_test_rmse = sum_test_rmse + evaluate(y_pred_test, y_test).rmse

    return sum_criterion / max_iteration, sum_test_rmse / max_iteration


def weighted_cost(y_pred_train, y_pred_validation, weight=5):
    train_criterion = weight * abs(y_pred_train - y_pred_validation) + (y_pred_train + y_pred_validation)
    return train_criterion


def dataset_split(x, y):
    """
    split the data
    :param x:
    :param y:
    :return:
    """
    x_tv, x_test, y_tv, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    x_train, x_validation, y_train, y_validation = train_test_split(x_tv, y_tv, test_size=0.2)
    return x_test, x_train, x_tv, x_validation, y_test, y_train, y_tv, y_validation


def decision_tree_regression(x, y, max_depth):
    """
    function that applies decision tree regression model to predict the targets
    :param x:
    :param y:
    :param max_depth:
    :return:
    """
    decision_tree_reg = DecisionTreeRegressor(max_depth=max_depth)
    sum_criterion = 0
    sum_test_rmse = 0

    for i in range(max_iteration):
        x_test, x_train, x_tv, x_validation, y_test, y_train, y_tv, y_validation = dataset_split(x, y)
        decision_tree_reg.fit(x_train, y_train)
        y_pred_train = evaluate(decision_tree_reg.predict(x_train), y_train).rmse
        y_pred_validation = evaluate(decision_tree_reg.predict(x_validation), y_validation).rmse
        train_criterion = weighted_cost(y_pred_train, y_pred_validation)

        decision_tree_reg.fit(x_tv, y_tv)
        y_pred_test = decision_tree_reg.predict(x_test)
        sum_criterion = sum_criterion + train_criterion
        sum_test_rmse = sum_test_rmse + evaluate(y_pred_test, y_test).rmse

    return sum_criterion / max_iteration, sum_test_rmse / max_iteration


def support_vector_regression(x, y):
    """
    function that applies support vector regression model to predict the targets
    :param x:
    :param y:
    :return:
    """
    linear_svr = SVR(kernel='rbf', epsilon=10)
    sum_criterion = 0
    sum_test_rmse = 0

    for i in range(max_iteration):
        x_test, x_train, x_tv, x_validation, y_test, y_train, y_tv, y_validation = dataset_split(x, y)
        linear_svr.fit(x_train, y_train)
        y_pred_train = evaluate(linear_svr.predict(x_train), y_train).rmse
        y_pred_validation = evaluate(linear_svr.predict(x_validation), y_validation).rmse
        train_criterion = weighted_cost(y_pred_train, y_pred_validation)

        linear_svr.fit(x_tv, y_tv)
        y_pred_test = linear_svr.predict(x_test)
        sum_criterion = sum_criterion + train_criterion
        sum_test_rmse = sum_test_rmse + evaluate(y_pred_test, y_test).rmse

    return sum_criterion / max_iteration, sum_test_rmse / max_iteration


def gradient_boosting_regression(x, y):
    """
    function that applies gradient boosting regression model to predict the targets
    :param x:
    :param y:
    :return:
    """
    x_test, x_train, x_tv, x_validation, y_test, y_train, y_tv, y_validation = dataset_split(x, y)

    gbdt = GradientBoostingRegressor(
        loss='ls'
        , learning_rate=0.05
        , n_estimators=150
        , subsample=1
        , min_samples_split=2
        , min_samples_leaf=1
        , max_depth=3
        , init=None
        , random_state=None
        , max_features=None
        , alpha=0.9
        , verbose=0
        , max_leaf_nodes=None
        , warm_start=False
    )
    sum_criterion = 0
    sum_test_rmse = 0

    for i in range(max_iteration):
        x_test, x_train, x_tv, x_validation, y_test, y_train, y_tv, y_validation = dataset_split(x, y)
        gbdt.fit(x_train, y_train)
        y_pred_train = evaluate(gbdt.predict(x_train), y_train).rmse
        y_pred_validation = evaluate(gbdt.predict(x_validation), y_validation).rmse
        train_criterion = weighted_cost(y_pred_train, y_pred_validation)

        gbdt.fit(x_tv, y_tv)
        y_pred_test = gbdt.predict(x_test)
        sum_criterion = sum_criterion + train_criterion
        sum_test_rmse = sum_test_rmse + evaluate(y_pred_test, y_test).rmse

    return sum_criterion / max_iteration, sum_test_rmse / max_iteration


def random_forest_regression(x, y, max_depth):
    """
    function that applies random forest regression model to predict the targets
    :param x:
    :param y:
    :param max_depth:
    :return:
    """
    rf = RandomForestRegressor(max_depth=max_depth)

    sum_criterion = 0
    sum_test_rmse = 0

    for i in range(max_iteration):
        x_test, x_train, x_tv, x_validation, y_test, y_train, y_tv, y_validation = dataset_split(x, y)
        rf.fit(x_train, y_train)
        y_pred_train = evaluate(rf.predict(x_train), y_train).rmse
        y_pred_validation = evaluate(rf.predict(x_validation), y_validation).rmse
        train_criterion = weighted_cost(y_pred_train, y_pred_validation)

        rf.fit(x_tv, y_tv)
        y_pred_test = rf.predict(x_test)
        sum_criterion = sum_criterion + train_criterion
        sum_test_rmse = sum_test_rmse + evaluate(y_pred_test, y_test).rmse

    return sum_criterion / max_iteration, sum_test_rmse / max_iteration


def ridge_regression(x, y):
    """
    function that applies ridge regression model to predict the targets
    :param x:
    :param y:
    :return:
    """
    ridge_model = Ridge(alpha=1.0, fit_intercept=True, solver='auto', copy_X=True)

    sum_criterion = 0
    sum_test_rmse = 0

    for i in range(max_iteration):
        x_test, x_train, x_tv, x_validation, y_test, y_train, y_tv, y_validation = dataset_split(x, y)
        ridge_model.fit(x_train, y_train)
        y_pred_train = evaluate(ridge_model.predict(x_train), y_train).rmse
        y_pred_validation = evaluate(ridge_model.predict(x_validation), y_validation).rmse
        train_criterion = weighted_cost(y_pred_train, y_pred_validation)

        ridge_model.fit(x_tv, y_tv)
        y_pred_test = ridge_model.predict(x_test)
        sum_criterion = sum_criterion + train_criterion
        sum_test_rmse = sum_test_rmse + evaluate(y_pred_test, y_test).rmse

    return sum_criterion / max_iteration, sum_test_rmse / max_iteration


def cnn_regression(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    cnn_model = Sequential()

    cnn_model.add(Conv1D(
        filters=32,
        kernel_size=2,
        strides=1,
        padding='same',
        data_format='channels_first'
    ))
    cnn_model.add(Activation('relu'))

    cnn_model.add(MaxPooling1D(
        pool_size=2,
        strides=2,
        padding='same',
        data_format='channels_first'
    ))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Conv1D(
        64, 2, strides=1, padding='same', data_format='channels_first'
    ))
    cnn_model.add(Activation('relu'))

    cnn_model.add(MaxPooling1D(2, 2, 'same', data_format='channels_first'))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(128))
    cnn_model.add(Activation('relu'))

    # Fully connected layer 2 to shape (1) for linear result
    cnn_model.add(Dense(1, kernel_initializer='normal'))
    cnn_model.add(Activation('linear'))

    # Another way to define your optimizer
    adam = Adam(lr=1e-4)

    # We add metrics to get more results you want to see
    cnn_model.compile(optimizer=adam,
                      loss=losses.mean_absolute_error,
                      metrics=['accuracy'])

    sum_criterion = 0
    sum_test_rmse = 0

    for i in range(max_iteration):
        x_test, x_train, x_tv, x_validation, y_test, y_train, y_tv, y_validation = dataset_split(x, y)
        columns = x.shape[1]
        x_train = x_train.reshape((int(len(x_train)), columns, 1))
        x_test = x_test.reshape((int(len(x_test)), columns, 1))
        x_tv = x_tv.reshape((int(len(x_tv)), columns, 1))
        x_validation = x_validation.reshape((int(len(x_validation)), columns, 1))

        cnn_model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=0)
        y_pred_train = evaluate(cnn_model.predict(x_train), y_train).rmse
        y_pred_validation = evaluate(cnn_model.predict(x_validation), y_validation).rmse
        train_criterion = weighted_cost(y_pred_train, y_pred_validation)

        cnn_model.fit(x_tv, y_tv, epochs=10, batch_size=64, verbose=0)
        y_pred_test = cnn_model.predict(x_test)

        sum_criterion = sum_criterion + train_criterion
        sum_test_rmse = sum_test_rmse + evaluate(y_pred_test, y_test).rmse

    return sum_criterion / max_iteration, sum_test_rmse / max_iteration


def rnn_regression(x, y):
    """
    applies rnn model to predict the target
    :return:
    """
    rnn_model = Sequential()

    # build a LSTM RNN
    rnn_model.add(LSTM(32))
    # add output layer
    rnn_model.add(Dense(OUTPUT_SIZE))
    adam = Adam(LR)
    rnn_model.compile(optimizer=adam,
                      loss=losses.mean_squared_error, )

    sum_criterion = 0
    sum_test_rmse = 0

    for i in range(max_iteration):
        x_test, x_train, x_tv, x_validation, y_test, y_train, y_tv, y_validation = dataset_split(x, y)

        columns = x.shape[1]
        x_train = x_train.reshape((int(len(x_train)), columns, 1))
        x_test = x_test.reshape((int(len(x_test)), columns, 1))
        rnn_model.fit(x_train, y_train, batch_size=50, verbose=0)
        y_pred_train = evaluate(rnn_model.predict(x_train), y_train).rmse
        y_pred_validation = evaluate(rnn_model.predict(x_validation), y_validation).rmse
        train_criterion = weighted_cost(y_pred_train, y_pred_validation)

        rnn_model.fit(x_tv, y_tv, batch_size=50)
        y_pred_test = rnn_model.predict(x_test)

        sum_criterion = sum_criterion + train_criterion
        sum_test_rmse = sum_test_rmse + evaluate(y_pred_test, y_test).rmse

    return sum_criterion / max_iteration, sum_test_rmse / max_iteration


def evaluate(y_pred, y_test):
    # calculate MSE
    mse = metrics.mean_squared_error(y_test, y_pred)
    # print("MSE:", mse)
    # calculate RMSE
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    # print("RMSE:", rmse)
    # calculate MDE
    mde = metrics.median_absolute_error(y_test, y_pred)
    # print("MDE:", mde)
    # calculate MAE
    mae = metrics.mean_squared_error(y_test, y_pred)
    # print("MAE:", mae)
    # calculate r2 score
    r_score = metrics.r2_score(y_test, y_pred)
    # print("R SCORE:", r_score)

    return Evaluation(mse, rmse, mde, mae, r_score)
