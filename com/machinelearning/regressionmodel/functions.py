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


def linear_regression(x, y):
    """
     function to apply linear regression model to predict the targets
    :param x:
    :param y:
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)

    y_pred = lin_reg.predict(x_test)
    return evaluate(y_pred, y_test)


def decision_tree_regression(x, y, max_depth):
    """
    function that applies decision tree regression model to predict the targets
    :param x:
    :param y:
    :param max_depth:
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

    decision_tree_reg = DecisionTreeRegressor(max_depth=max_depth)
    decision_tree_reg.fit(x_train, y_train)

    y_pred = decision_tree_reg.predict(x_test)
    return evaluate(y_pred, y_test)


def support_vector_regression(x, y):
    """
    function that applies support vector regression model to predict the targets
    :param x:
    :param y:
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

    linear_svr = SVR(kernel='linear')
    linear_svr.fit(x_train, y_train.ravel())
    y_pred = linear_svr.predict(x_test)
    return evaluate(y_pred, y_test)


def gradient_boosting_regression(x, y):
    """
    function that applies gradient boosting regression model to predict the targets
    :param x:
    :param y:
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

    gbdt = GradientBoostingRegressor(
        loss='ls'
        , learning_rate=0.01
        , n_estimators=100
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
    gbdt.fit(x_train, y_train)
    y_pred = gbdt.predict(x_test)
    return evaluate(y_pred, y_test)


def random_forest_regression(x, y, max_depth):
    """
    function that applies random forest regression model to predict the targets
    :param x:
    :param y:
    :param max_depth:
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    rf = RandomForestRegressor(max_depth=max_depth)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    return evaluate(y_pred, y_test)


def ridge_regression(x, y):
    """
    function that applies ridge regression model to predict the targets
    :param x:
    :param y:
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    ridge_model = Ridge(alpha=1.0, fit_intercept=True, solver='auto', copy_X=True)
    ridge_model.fit(x_train, y_train)
    y_pred = ridge_model.predict(x_test)
    return evaluate(y_pred, y_test)


def cnn_regression(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    columns = x.shape[1]
    # (-1,5,1)
    x_train = x_train.reshape((int(len(x_train)), columns, 1))
    x_test = x_test.reshape((int(len(x_test)), columns, 1))
    model = Sequential()

    model.add(Conv1D(
        filters=32,
        kernel_size=2,
        strides=1,
        padding='same',
        data_format='channels_first'
    ))
    model.add(Activation('relu'))

    model.add(MaxPooling1D(
        pool_size=2,
        strides=2,
        padding='same',
        data_format='channels_first'
    ))
    model.add(Dropout(0.2))
    model.add(Conv1D(
        64, 2, strides=1, padding='same', data_format='channels_first'
    ))
    model.add(Activation('relu'))

    model.add(MaxPooling1D(2, 2, 'same', data_format='channels_first'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))

    # Fully connected layer 2 to shape (1) for linear result
    model.add(Dense(1, kernel_initializer='normal'))
    model.add(Activation('linear'))

    # Another way to define your optimizer
    adam = Adam(lr=1e-4)

    # We add metrics to get more results you want to see
    model.compile(optimizer=adam,
                  loss=losses.mean_squared_error,
                  metrics=['accuracy'])

    print('Training ------------')
    # Another way to train the model
    model.fit(x_train, y_train, epochs=10, batch_size=64)

    print('\nTesting ------------')
    y_pred = model.predict(x_test)

    # Evaluate the model with the metrics we defined earlier
    loss, accuracy = model.evaluate(x_test, y_test)
    print('\ntest loss: ', loss)
    print('\ntest accuracy: ', accuracy)

    return evaluate(y_pred, y_test)


def rnn_regression(x, y):
    """
    applies rnn model to predict the target
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    columns = x.shape[1]
    x_train = x_train.reshape((int(len(x_train)), columns, 1))
    x_test = x_test.reshape((int(len(x_test)), columns, 1))

    # y_train = y_train.reshape((int(len(y_train)), 1, 1))
    # y_test = y_test.reshape((int(len(y_test)), 1, 1))

    model = Sequential()

    # build a LSTM RNN
    model.add(LSTM(32))
    # add output layer
    model.add(Dense(OUTPUT_SIZE))
    adam = Adam(LR)
    model.compile(optimizer=adam,
                  loss=losses.mean_squared_error, )

    print('Training ------------')
    model.fit(x_train, y_train, batch_size=50)
    y_pred = model.predict(x=x_test)
    return evaluate(y_pred, y_test)


def evaluate(y_pred, y_test):
    # calculate MSE
    mse = metrics.mean_squared_error(y_test, y_pred)
    print("MSE:", mse)
    # calculate RMSE
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print("RMSE:", rmse)
    # calculate MDE
    mde = metrics.median_absolute_error(y_test, y_pred)
    print("MDE:", mde)
    # calculate MAE
    mae = metrics.mean_squared_error(y_test, y_pred)
    print("MAE:", mae)
    # calculate r2 score
    r_score = metrics.r2_score(y_test, y_pred)
    print("R SCORE:", r_score)

    return Evaluation(mse, rmse, mde, mae, r_score)
