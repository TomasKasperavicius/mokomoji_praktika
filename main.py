from pmdarima import auto_arima
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool,cpu_count
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from statsmodels.tsa.seasonal import seasonal_decompose
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import set_random_seed
from itertools import product
from sklearn.multioutput import MultiOutputRegressor
import tensorflow as tf


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_true = y_true + 1
    y_pred = y_pred + 1

    percentage_diff = np.abs((y_true - y_pred) / y_true)

    mape = np.mean(percentage_diff) * 100
    return mape

def read_data(file_name):
    if os.path.exists(file_name):
        df = pd.read_csv(file_name, sep=',')
        df['Laikotarpis'] = pd.to_datetime(df['Laikotarpis'])
        df.set_index('Laikotarpis', inplace=True)
        df.index.freq = 'MS'
        return df
    else:
        print(f"The file '{file_name}' does not exist.")

def plot_error_values(error_values_sarima, error_values_svr, error_values_lstm, error_values_sarima_svr):
    error_values = [error_values_svr, error_values_sarima, error_values_lstm, error_values_sarima_svr]
    model_names = ['SVR', 'ARIMA', 'LSTM', 'ARIMA+SVR']
    error_names = ['RMSE','MAE', 'MAPE']
    titles = ['Vidutinė kvadratinė paklaida','Vidutinė absoliuti paklaida','Vidutinė absoliuti procentinė paklaida']
    colors=['green', 'orange', 'blue', 'red']
    _, axs = plt.subplots(3, 1, figsize=(14, 7))
    for k,error_name in enumerate(error_names):
        all_errors = []
        for i, error_value in enumerate(error_values):      
            all_errors.append(error_value[f'{error_name}_train'])
            axs[k].barh(i, [error_value[f'{error_name}_train']], color=colors[i])
            axs[k].set_yticks(range(len(model_names)))
            axs[k].set_yticklabels(model_names, fontsize=14)
            axs[k].set_title(f'{titles[k]}', fontsize=14)
            axs[k].tick_params(axis='x', which='both', labelsize=12)
            
        for j, value in enumerate(all_errors):
            axs[k].text(value, j, f'{value:.2f}', ha='left', va='center', color='black', fontsize=14)
    plt.tick_params(axis='both', which='both', labelsize=12)
    plt.suptitle('Mokymo duomenų aibė', fontsize=14)
    plt.tight_layout()
    plt.show()
    _, axs = plt.subplots(3, 1, figsize=(14, 7))
    for k,error_name in enumerate(error_names):
        all_errors = []
        for i, error_value in enumerate(error_values):      
            all_errors.append(error_value[f'{error_name}_test'])
            axs[k].barh(i, [error_value[f'{error_name}_test']], color=colors[i])
            axs[k].set_yticks(range(len(model_names)))
            axs[k].set_yticklabels(model_names, fontsize=14)
            axs[k].set_title(f'{titles[k]}', fontsize=14)
            axs[k].tick_params(axis='x', which='both', labelsize=12)
            
        for j, value in enumerate(all_errors):
            axs[k].text(value, j, f'{value:.2f}', ha='left', va='center', color='black', fontsize=14)
    plt.tick_params(axis='both', which='both', labelsize=12)
    plt.suptitle('Testavimo duomenų aibė', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_results(original, forecasts, labels, title, ylabel, start, end, month):
    date_labels = pd.date_range(start=start, end=end, freq='MS')
    if not SEASONAL:
        date_labels = date_labels[date_labels.month == month]
        plt.plot(date_labels, original.values[LAG_COUNT:],
             label='Originalios reikšmės', color="#800080", marker='o')
        plt.xticks(date_labels, [date.strftime('%Y-%m')
                for date in date_labels])
        plt.xticks(rotation=90, fontsize=12)
        plt.title(title, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.xlabel('Laikotarpis', fontsize=16)

        for i in range(len(forecasts)):
            forecasts[i] = np.concatenate((forecasts[i][0], forecasts[i][1]))
            if labels[i] == 'ARIMA':
                plt.plot(date_labels, forecasts[i][LAG_COUNT:], label=labels[i],
                        linestyle='--', color=plt.cm.tab10(i), marker='o')
                continue
            plt.plot(date_labels, forecasts[i], label=labels[i],
                    linestyle='--', color=plt.cm.tab10(i), marker='o')
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.grid()
        plt.show()
    else:
        values = original.values[:-TEST_SIZE_IN_YEARS*12]
        plt.plot(date_labels, values[LAG_COUNT:],
                label='Originalios reikšmės', color="#800080", marker='o')
        plt.xticks(date_labels, [date.strftime('%Y-%m')
                for date in date_labels])
        plt.xticks(rotation=90, fontsize=12)
        plt.title(title, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.xlabel('Laikotarpis', fontsize=16)

        for i in range(len(forecasts)):
            if labels[i] == 'ARIMA':
                plt.plot(date_labels, forecasts[i][0][LAG_COUNT:], label=labels[i],
                        linestyle='--', color=plt.cm.tab10(i), marker='o')
                continue
            plt.plot(date_labels, forecasts[i][0], label=labels[i],
                    linestyle='--', color=plt.cm.tab10(i), marker='o')
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.grid()
        plt.show()
        start = (pd.to_datetime(end) + pd.DateOffset(months=1)).strftime('%Y-%m-%d')
        end = (pd.to_datetime(end) +
            pd.DateOffset(years=TEST_SIZE_IN_YEARS)).strftime('%Y-%m-%d')
        date_labels = pd.date_range(start=start, end=end, freq='MS')
        values = original.values[-TEST_SIZE_IN_YEARS*12:]
        
        plt.plot(date_labels, values,
                label='Originalios reikšmės', color="#800080", marker='o')
        plt.xticks(date_labels, [date.strftime('%Y-%m')
                for date in date_labels])
        plt.xticks(rotation=45, fontsize=12)
        plt.title(title, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.xlabel('Laikotarpis', fontsize=16)
        for i in range(len(forecasts)):
            if labels[i] == 'SARIMA':
                plt.plot(date_labels, forecasts[i][1], label=labels[i],
                        linestyle='--', color=plt.cm.tab10(i), marker='o')
                continue
            plt.plot(date_labels, forecasts[i][1], label=labels[i],
                    linestyle='--', color=plt.cm.tab10(i), marker='o')
        plt.legend(fontsize=12)
        plt.grid()
        plt.tight_layout()
        plt.show()

def analyze_data_patterns(df, label):
    decomposition = seasonal_decompose(
        df, model='additive', period=12, extrapolate_trend='freq')
    trend, seasonal, residual = decomposition.trend, decomposition.seasonal, decomposition.resid
    labels = ['Originalios reikšmės',
              'Tendencija', 'Sezoniškumas', 'Triukšmas']
    data = [df[label], trend, seasonal.head(12), residual]
    fig,  axs = plt.subplots(4, 1, figsize=(
        20, 18), gridspec_kw={'hspace': 0.5})
    fig.suptitle("Sezoninė dekompozicija")
    date_labels = pd.date_range(
        start=df.index.min(), end=df.index.max(), freq='MS')
    for i in range(len(data)):
        tick_positions = range(0, len(date_labels), 12)
        if i == 2:
            axs[i].plot(['Sausis',
                         'Vasaris',
                         'Kovas',
                         'Balandis',
                         'Gegužė',
                         'Birželis',
                         'Liepa',
                         'Rugpjūtis',
                         'Rugsėjis',
                         'Spalis',
                         'Lapkritis',
                         'Gruodis'], data[i])
            axs[i].legend(loc='upper right', fontsize=12)
            axs[i].set_title(labels[i], fontsize=12)
        else:
            axs[i].plot(date_labels, data[i])
            axs[i].set_xticks(date_labels[tick_positions])
            axs[i].set_xticklabels([date.strftime('%Y')
                                    for date in date_labels[tick_positions]])
            axs[i].set_title(labels[i], fontsize=12)
        axs[i].set_xlabel('Laikotarpis', fontsize=12)
        axs[i].set_ylabel(label, fontsize=12)
        axs[i].tick_params(axis='both', which='both', labelsize=12)
        
        axs[i].grid()
    plt.tight_layout()
    plt.show()

def evaluate_performance(forecast, real):
    mae = mean_absolute_error(real, forecast)
    rmse = np.sqrt(mean_squared_error(real, forecast))
    mape = mean_absolute_percentage_error(real, forecast)
    print(
        f"Mean absolute error: {mae}, Root mean squared error: {rmse}, Mean absolute percentage error: {mape}")
    return mae, rmse, mape

def perform_ARIMA(train, test, label):
    if SEASONAL:
        model = auto_arima(train,
                        m=12,
                        start_p=1,
                        start_q=1,
                        trace=True,
                        seasonal=True,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True,
                        n_fits=100, n_jobs=-1)
    else:
        model = auto_arima(train,
                        m=1,
                        start_p=1,
                        start_q=1,
                        seasonal=False,
                        trace=True,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True,
                        n_fits=100, n_jobs=-1)
    model.fit(train)
    test_forecast, conf_int = model.predict(
        n_periods=len(test), return_conf_int=True)
    train_forecast = model.predict_in_sample()
    # print(model.summary())
    # print(conf_int)
    error_values_sarima = {}
    print("Training data evaluation:")
    error_values_sarima["MAE_train"], error_values_sarima["RMSE_train"], error_values_sarima["MAPE_train"] = evaluate_performance(
        train_forecast, train[label].values)
    print("Testing data evaluation:")
    error_values_sarima["MAE_test"], error_values_sarima["RMSE_test"], error_values_sarima["MAPE_test"] = evaluate_performance(
        test_forecast, test[label].values)
    return np.array(train_forecast), np.array(test_forecast), error_values_sarima

def find_best_SVR_parameters(parameter_combinations, train_X, train_Y, validation_X, validation_Y, scalerY):
    set_random_seed(10)
    np.random.seed(10)
    tf.random.set_seed(10)
    parameter_grid = {"C": [0.9,1, 1.1],
                          "gamma": [0.1, 0.05, 0.01,0.001,0.0001, 'auto', 'scale'],
                          "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                          "epsilon": [0.1, 0.05, 0.01],
                          "degree": [2, 3, 4, 5]}
    best_mae = float("inf")
    best_params = None
    for params in parameter_combinations:
        param_dict = dict(zip(parameter_grid.keys(), params))
        svr = SVR(kernel=param_dict["kernel"], C=param_dict["C"], gamma=param_dict['gamma'],
                  epsilon=param_dict['epsilon'], degree=param_dict['degree'])
        svr = MultiOutputRegressor(svr)
        svr.fit(train_X, train_Y)
        validation_forecast = svr.predict(validation_X)
        validation_forecast = scalerY.inverse_transform(
            validation_forecast)
        mae = mean_absolute_error(validation_Y, validation_forecast)
        if mae < best_mae:
            best_mae = mae
            best_params = params
    return [{"C": best_params[0],
             "gamma": best_params[1],
             "kernel": best_params[2],
             "epsilon": best_params[3],
            "degree": best_params[4]}, best_mae]

def perform_SVR(non_overlapping_X, non_overlapping_Y, overlapping_X, best_params=None):
    set_random_seed(10)
    np.random.seed(10)
    tf.random.set_seed(10)
    scalerX = MinMaxScaler(feature_range=(0, 1))
    scalerY = MinMaxScaler(feature_range=(0, 1))
    non_overlapping_X = scalerX.fit_transform(non_overlapping_X)
    non_overlapping_Y = scalerY.fit_transform(non_overlapping_Y)
    test_X, test_Y = non_overlapping_X[-TEST_SIZE_IN_YEARS:
                                       ], non_overlapping_Y[-TEST_SIZE_IN_YEARS:]
    test_size = len(test_X)
    train_X, train_Y = non_overlapping_X[:-
                                         TEST_SIZE_IN_YEARS], non_overlapping_Y[:-TEST_SIZE_IN_YEARS]
    train_size = len(train_X)
    validation_X, validation_Y = train_X[train_size -
                                         test_size:], train_Y[train_size-test_size:]
    if best_params is None:
        new_train_X, new_train_Y = train_X[:train_size -
                                           test_size], train_Y[:train_size-test_size]
        parameter_grid = {"C": [0.9,1, 1.1],
                          "gamma": [0.1, 0.05, 0.01,0.001,0.0001, 'auto', 'scale'],
                          "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                          "epsilon": [0.1, 0.05, 0.01],
                          "degree": [2, 3, 4, 5]}
        param_combinations = list(product(*parameter_grid.values()))
        batch_size = len(param_combinations)//cpu_count()  # 8 cores
        with Pool() as pool:
            batches = [param_combinations[i:i+batch_size]
                       for i in range(0, len(param_combinations), batch_size)]
            results = pool.starmap(find_best_SVR_parameters, [(
                batch, new_train_X, new_train_Y, validation_X, validation_Y, scalerY) for batch in batches])
        best_params = min(results, key=lambda x: x[1])[0]
        print(best_params)
    svr = SVR(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'],
              epsilon=best_params['epsilon'], degree=best_params['degree'])
    svr = MultiOutputRegressor(svr)
    svr.fit(train_X, train_Y)
    train_forecast = svr.predict(train_X)
    train_forecast = scalerY.inverse_transform(
        train_forecast).reshape(-1, 1)
    test_forecast = svr.predict(test_X)
    test_forecast = scalerY.inverse_transform(
        test_forecast).reshape(-1, 1)
    error_values_svr = {}

    print("Training data evaluation:")
    error_values_svr["MAE_train"], error_values_svr["RMSE_train"], error_values_svr["MAPE_train"] = evaluate_performance(
        train_forecast, scalerY.inverse_transform(train_Y).reshape(-1, 1))
    print("Testing data evaluation:")
    error_values_svr["MAE_test"], error_values_svr["RMSE_test"], error_values_svr["MAPE_test"] = evaluate_performance(
        test_forecast, scalerY.inverse_transform(test_Y).reshape(-1, 1))

    train_forecasts = []
    test_forecasts = []
    for i in range(len(overlapping_X)-TEST_SIZE_IN_YEARS):
        result = svr.predict(scalerX.transform([overlapping_X[i]]))
        train_forecasts.append(
            scalerY.inverse_transform(result))
    result = svr.predict(scalerX.transform(
        overlapping_X[-TEST_SIZE_IN_YEARS:]))
    test_forecasts.append(
        scalerY.inverse_transform(result))
    return np.array(train_forecasts).flatten(), np.array(test_forecasts).flatten(), error_values_svr

def preprocess_data(data, lag_count, forecasting_period, SPLIT=0.8):
    overlapping_X = []
    overlapping_Y = []
    non_overlapping_X = []
    non_overlapping_Y = []
    for i in range(0, len(data)-lag_count, forecasting_period):
        overlapping_X.append(data.iloc[i:i + lag_count])
        overlapping_Y.append(
            data.iloc[i + lag_count:i + lag_count+forecasting_period])
    for i in range(0, len(data)-lag_count, SLIDING_WINDOW_SIZE):
        non_overlapping_X.append(data.iloc[i:i + lag_count])
        non_overlapping_Y.append(
            data.iloc[i+lag_count:i + lag_count+forecasting_period])
    overlapping_X = np.array(overlapping_X).reshape(-1, lag_count)
    overlapping_Y = np.array(overlapping_Y).reshape(-1, forecasting_period)
    non_overlapping_X = np.array(non_overlapping_X).reshape(-1, lag_count)
    non_overlapping_Y = np.array(
        non_overlapping_Y).reshape(-1, forecasting_period)
    return overlapping_X, overlapping_Y, non_overlapping_X, non_overlapping_Y

def perform_LSTM(non_overlapping_X, non_overlapping_Y, overlapping_X, model_name, parameters, use_model=False):
    set_random_seed(10)
    np.random.seed(10)
    tf.random.set_seed(10)
    scalerX = MinMaxScaler(feature_range=(-1, 1))
    scalerY = MinMaxScaler(feature_range=(-1, 1))
    non_overlapping_X = scalerX.fit_transform(non_overlapping_X)
    non_overlapping_Y = scalerY.fit_transform(non_overlapping_Y)
    test_X, test_Y = non_overlapping_X[-TEST_SIZE_IN_YEARS:
                                       ], non_overlapping_Y[-TEST_SIZE_IN_YEARS:]
    test_size = len(test_X)
    train_X, train_Y = non_overlapping_X[:-
                                         TEST_SIZE_IN_YEARS], non_overlapping_Y[:-TEST_SIZE_IN_YEARS]
    train_size = len(train_X)
    validation_X, validation_Y = train_X[train_size -
                                         test_size:], train_Y[train_size-test_size:]
    train_X = train_X.reshape(
        train_X.shape[0], train_X.shape[1], 1)
    test_X = test_X.reshape(
        test_X.shape[0], test_X.shape[1], 1)
    validation_X = validation_X.reshape(
        validation_X.shape[0], validation_X.shape[1], 1)
    
    model = None
    if use_model and os.path.exists(f'lstm_model_{model_name}/'):
        model = load_model(f'lstm_model_{model_name}/')
    else:
        model = Sequential()
        model.add(LSTM(units=parameters['units'], dropout=parameters['dropout'],
                       activation=parameters['activation'], input_shape=(test_X.shape[1], 1)))
        model.add(Dense(units=FORECASTING_PERIOD, activation='tanh'))
        checkpoint = ModelCheckpoint(
            f'lstm_model_{model_name}/', save_best_only=True, monitor='val_loss')
        model.compile(optimizer=Adam(learning_rate=0.005), loss='mean_squared_error')
        model.fit(train_X, train_Y,
                  epochs=parameters['epochs'],
                  batch_size=parameters['batch_size'],
                  validation_data=(validation_X, validation_Y),
                  callbacks=[checkpoint],
                 workers=cpu_count(),use_multiprocessing=True)
        model = load_model(f'lstm_model_{model_name}/')
    train_forecast = model.predict(train_X, verbose=0)
    train_forecast = scalerY.inverse_transform(train_forecast)
    test_forecast = model.predict(test_X, verbose=0)
    test_forecast = scalerY.inverse_transform(test_forecast)
    error_values_lstm = {}

    print("Training data evaluation:")
    error_values_lstm["MAE_train"], error_values_lstm["RMSE_train"], error_values_lstm["MAPE_train"] = evaluate_performance(
        train_forecast, scalerY.inverse_transform(train_Y))
    print("Testing data evaluation:")
    error_values_lstm["MAE_test"], error_values_lstm["RMSE_test"], error_values_lstm["MAPE_test"] = evaluate_performance(
        test_forecast, scalerY.inverse_transform(test_Y))
    train_forecast = []
    test_forecast = []
    train_forecast = model.predict(scalerX.transform(overlapping_X[:-TEST_SIZE_IN_YEARS]).reshape(
        len(overlapping_X)-TEST_SIZE_IN_YEARS, train_X.shape[1], 1), verbose=0)
    test_forecast = model.predict(scalerX.transform(overlapping_X[-TEST_SIZE_IN_YEARS:]).reshape(
        TEST_SIZE_IN_YEARS, train_X.shape[1], 1), verbose=0)
    train_forecast = np.array(
        scalerY.inverse_transform(train_forecast)).reshape(-1, 1)
    test_forecast = np.array(
        scalerY.inverse_transform(test_forecast)).reshape(-1, 1)
    return train_forecast.flatten(), test_forecast.flatten(), error_values_lstm


def find_best_SVR_ARIMA_parameters(parameter_combinations,  train_X, train_Y, validation_X, validation_Y, scalerY):
    set_random_seed(10)
    np.random.seed(10)
    tf.random.set_seed(10)
    parameter_grid = {"C": [0.9,1, 1.1],
                          "gamma": [0.1, 0.05, 0.01,0.001,0.0001, 'auto', 'scale'],
                          "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                          "epsilon": [0.1, 0.05, 0.01],
                          "degree": [2, 3, 4, 5]}
    best_mae = float("inf")
    best_params = None
    for params in parameter_combinations:
        param_dict = dict(zip(parameter_grid.keys(), params))
        svr = SVR(kernel=param_dict["kernel"], C=param_dict["C"], gamma=param_dict['gamma'],
                  epsilon=param_dict['epsilon'], degree=param_dict['degree'])
        svr = MultiOutputRegressor(svr)
        svr.fit(train_X, train_Y)
        validation_forecast = svr.predict(validation_X)
        validation_forecast = scalerY.inverse_transform(
            validation_forecast)
        mae = mean_absolute_error(validation_Y, validation_forecast)
        if mae < best_mae:
            best_mae = mae
            best_params = params
    return [{"C": best_params[0],
             "gamma": best_params[1],
             "kernel": best_params[2],
             "epsilon": best_params[3],
            "degree": best_params[4]}, best_mae]


def perform_SARIMA_SVR(non_overlapping_X, non_overlapping_Y, overlapping_X, best_params=None):
    set_random_seed(10)
    np.random.seed(10)
    tf.random.set_seed(10)
    scalerX = MinMaxScaler(feature_range=(0, 1))
    scalerY = MinMaxScaler(feature_range=(0, 1))
    non_overlapping_X = scalerX.fit_transform(non_overlapping_X)
    non_overlapping_Y = scalerY.fit_transform(non_overlapping_Y)
    test_X, test_Y = non_overlapping_X[-TEST_SIZE_IN_YEARS:
                                       ], non_overlapping_Y[-TEST_SIZE_IN_YEARS:]
    test_size = len(test_X)
    train_X, train_Y = non_overlapping_X[:-
                                         TEST_SIZE_IN_YEARS], non_overlapping_Y[:-TEST_SIZE_IN_YEARS]
    train_size = len(train_X)
    validation_X, validation_Y = train_X[train_size -
                                         test_size:], train_Y[train_size-test_size:]

    if best_params is None:
        new_train_X, new_train_Y = train_X[:train_size -
                                           test_size], train_Y[:train_size-test_size]
        parameter_grid = {"C": [0.9,1, 1.1],
                          "gamma": [0.1, 0.05, 0.01,0.001,0.0001, 'auto', 'scale'],
                          "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                          "epsilon": [0.1, 0.05, 0.01],
                          "degree": [2, 3, 4, 5]}
        
        param_combinations = list(product(*parameter_grid.values()))
        batch_size = len(param_combinations)//cpu_count()  # 8 cores
        with Pool() as pool:
            batches = [param_combinations[i:i+batch_size]
                       for i in range(0, len(param_combinations), batch_size)]
            results = pool.starmap(find_best_SVR_ARIMA_parameters, [
                                   (batch, new_train_X, new_train_Y, validation_X, validation_Y, scalerY) for batch in batches])
        best_params = min(results, key=lambda x: x[1])[0]
    svr = SVR(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'],
              epsilon=best_params['epsilon'], degree=best_params['degree'])
    svr = MultiOutputRegressor(svr)
    svr.fit(train_X, train_Y)
    train_forecast = svr.predict(train_X)
    train_forecast = scalerY.inverse_transform(
        train_forecast).reshape(-1, 1)
    test_forecast = svr.predict(test_X)
    test_forecast = scalerY.inverse_transform(test_forecast).reshape(-1, 1)
    error_values_sarima_svr = {}
    print("Training data evaluation:")
    error_values_sarima_svr["MAE_train"], error_values_sarima_svr["RMSE_train"], error_values_sarima_svr["MAPE_train"] = evaluate_performance(
        train_forecast, scalerY.inverse_transform(train_Y).reshape(-1, 1))
    print("Testing data evaluation:")
    error_values_sarima_svr["MAE_test"], error_values_sarima_svr["RMSE_test"], error_values_sarima_svr["MAPE_test"] = evaluate_performance(
        test_forecast, scalerY.inverse_transform(test_Y).reshape(-1, 1))
    train_forecasts = []
    test_forecasts = []
    for i in range(len(overlapping_X)-TEST_SIZE_IN_YEARS):
        result = svr.predict(scalerX.transform([overlapping_X[i]]))
        train_forecasts.append(
            scalerY.inverse_transform(result))
    result = svr.predict(scalerX.transform(
        overlapping_X[-TEST_SIZE_IN_YEARS:]))
    test_forecasts.append(
        scalerY.inverse_transform(result))
    return np.array(train_forecasts).flatten(), np.array(test_forecasts).flatten(), error_values_sarima_svr

if __name__ == "__main__":
    label = 'Kiekis, vnt'

    df_snow = read_data('sniego_valytuvai.csv')
    df_cheese = read_data('suriai.csv')
    df_notebooks = read_data('sasiuviniai.csv')
    
    # plot_original(df_snow,'Sniego valytuvų eksportas į Latviją','Kiekis, vnt', 'Laikotarpis')
    # plot_original(df_cheese,'Visų rūšių trintų arba miltelių pavidalo sūrių eksportas į Latviją','Kiekis, kg', 'Laikotarpis')
    # plot_original(df_notebooks,'Sąsiuvinių eksportas į Latviją','Kiekis, kg', 'Laikotarpis')

    # plt.boxplot(df_notebooks)
    # plt.boxplot(df_cheese)
    # plt.boxplot(df_snow)
    # plt.show()

    # analyze_data_patterns(df_snow,'Kiekis, kg')
    # analyze_data_patterns(df_cheese)
    # analyze_data_patterns(df_notebooks)
    

    best_LSTM_snow = {'units': 12, 'dropout': 0.1,
                      'activation': 'tanh', 'epochs': 100, 'batch_size': 1}
    best_LSTM_notebook = {'units': 12, 'dropout': 0.1,
                          'activation': 'tanh', 'epochs': 100, 'batch_size': 1}
    best_LSTM_cheese = {'units': 12, 'dropout': 0.1,
                        'activation': 'tanh', 'epochs': 100, 'batch_size': 1}
    
    best_SVR_snow = {'C': 1.1, 'gamma': 'scale', 'kernel': 'poly', 'epsilon': 0.01, 'degree': 2}
    best_SVR_notebook = {'C': 1.1, 'gamma': 'scale', 'kernel': 'poly', 'epsilon': 0.01, 'degree': 3}
    best_SVR_cheese = {'C': 0.9, 'gamma': 'scale', 'kernel': 'poly', 'epsilon': 0.01, 'degree': 5}

    best_ARIMA_SVR_snow = {'C': 1.5, 'gamma': 'scale', 'kernel': 'poly', 'epsilon': 0.01, 'degree': 3}
    best_ARIMA_SVR_notebook = {'C': 0.9, 'gamma': 'scale', 'kernel': 'poly', 'epsilon': 0.01, 'degree': 3}
    best_ARIMA_SVR_cheese = {'C': 0.9, 'gamma': 'scale', 'kernel': 'poly', 'epsilon': 0.01, 'degree': 5}

    count = 0
    months = [12, 7, 12]
    TEST_SIZE_IN_YEARS = 2
    LAG_COUNT = 2
    SLIDING_WINDOW_SIZE = 1
    FORECASTING_PERIOD = 1
    SEASONAL = False
    years = LAG_COUNT//12 if SEASONAL else LAG_COUNT
    years2 = TEST_SIZE_IN_YEARS if SEASONAL else 0
    datasets = [[df_snow, 'Kiekis, vnt', 'Sniego valytuvų eksportas iš Lietuvos į Latviją',
                 best_SVR_snow,
                 best_LSTM_snow,
                 best_ARIMA_SVR_snow,
                 (df_snow.index.min() + pd.DateOffset(years=years)
                  ).strftime('%Y-%m-%d'),
                 (df_snow.index.max() - pd.DateOffset(years=years2)
                  ).strftime('%Y-%m-%d'),
                 'snow_cleaners'],
                [df_notebooks, 'Kiekis, kg', 'Sąsiuvinių eksportas iš Lietuvos į Latviją',
                 best_SVR_notebook,
                 best_LSTM_notebook,
                 best_ARIMA_SVR_notebook,
                 (df_notebooks.index.min() +
                  pd.DateOffset(years=years)).strftime('%Y-%m-%d'),
                 (df_notebooks.index.max() - pd.DateOffset(years=years2)
                  ).strftime('%Y-%m-%d'),
                 'notebooks'],
                [df_cheese, 'Kiekis, kg', 'Sūrių eksportas iš Lietuvos į Latviją',
                 best_SVR_cheese,
                 best_LSTM_cheese,
                 best_ARIMA_SVR_cheese,
                 (df_cheese.index.min() +
                  pd.DateOffset(years=years)).strftime('%Y-%m-%d'),
                 (df_cheese.index.max() - pd.DateOffset(years=years2)
                  ).strftime('%Y-%m-%d'),
                 'cheese']]
    for i in datasets:
        count = 2
        df = i[0] if SEASONAL else i[0][i[0].index.month == months[count]]
        forecasts = []
        overlapping_X, overlapping_Y, non_overlapping_X, non_overlapping_Y = preprocess_data(
            df, LAG_COUNT, FORECASTING_PERIOD)
    
        if SEASONAL:
            arima_train,  arima_test = df.iloc[:-
                                            TEST_SIZE_IN_YEARS*12], df.iloc[-TEST_SIZE_IN_YEARS*12:]
        else:
            arima_train,  arima_test = df.iloc[:-TEST_SIZE_IN_YEARS], df.iloc[-TEST_SIZE_IN_YEARS:]
        
        forecasts = []
        print("---ARIMA---")
        arima_forecast_train, arima_forecast_test, error_values_sarima = perform_ARIMA(
            arima_train, arima_test, i[1])#, i[3]
        forecasts.append([
            arima_forecast_train, arima_forecast_test])
        print("---SVR-----")
        train_forecast, test_forecast, error_values_svr = perform_SVR(
            non_overlapping_X, non_overlapping_Y, overlapping_X)#, i[3]
        forecasts.append([train_forecast, test_forecast])

        print("---LSTM----")
        train_forecast, test_forecast, error_values_lstm = perform_LSTM(
            non_overlapping_X, non_overlapping_Y, overlapping_X, i[8], i[4], use_model=False)
        forecasts.append([train_forecast, test_forecast])

        print("---ARIMA+SVR-----")
        arima_forecasts = np.concatenate(
            (arima_forecast_train, arima_forecast_test))
        arima_lag_forecasts = []
        for j in range(0, len(arima_forecasts)-LAG_COUNT, SLIDING_WINDOW_SIZE):
            arima_lag_forecasts.append(arima_forecasts[j:j + LAG_COUNT])
        arima_forecasts_new = []
        for j in range(0, len(arima_forecasts)-LAG_COUNT, SLIDING_WINDOW_SIZE):
            arima_forecasts_new.append(arima_forecasts[j:j + LAG_COUNT])
        index = len(non_overlapping_X)
        arima_forecasts_new = np.array(arima_forecasts_new)
        arima_lag_forecasts = np.array(arima_lag_forecasts)
        overlapping_X = np.hstack((overlapping_X, arima_forecasts_new))
        non_overlapping_X = np.hstack((non_overlapping_X, arima_lag_forecasts))
        train_forecast, test_forecast, error_values_sarima_svr = perform_SARIMA_SVR(
            non_overlapping_X, non_overlapping_Y, overlapping_X)#, i[5]
        forecasts.append([train_forecast, test_forecast])
        plot_error_values(error_values_sarima, error_values_svr,
                          error_values_lstm, error_values_sarima_svr)
        plot_results(df, forecasts, [
                     'ARIMA', 'SVR','LSTM', 'ARIMA+SVR'], i[2], i[1], i[6], i[7], months[count])
        count += 1