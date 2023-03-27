import os
import json
import math
import numpy as np
import pandas as pd
import pmdarima as pm
import seaborn as sns
import statsmodels.api as sm
from itertools import product
from flask_cors import CORS
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, session, send_file, url_for
from flask_paginate import Pagination, get_page_parameter
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.style.use('seaborn-darkgrid')


from functools import lru_cache
 
#*** Flask configuration
 
# Define folder to save uploaded files to process further
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
 
# Define allowed files (for this example I want only csv file)
ALLOWED_EXTENSIONS = {'csv'}
 
app = Flask(__name__, template_folder='templateFiles', static_folder='staticFiles')
# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = 'E:\Skripsi\Flask\hello_flask\Coba\Dataset'
CORS(app)
 
# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'

#Global Variable
df = '' 
hourly_data = ''
daily_data = ''
filled_dataset = ''
time_resampled_data = ''
already_run = False
already_run_tr = False
already_run_pd = False
already_run_ds = False
already_run_fe = False
already_run_ar1 = False
already_run_ar2 = False

#Fill Missing Function
def fillMissingFunct(param):
        param['Timestamp'] = [datetime.utcfromtimestamp(x) for x in param['Timestamp']] 
        param['Open'] = param['Open'].interpolate()
        param['Close'] = param['Close'].interpolate()
        param['Weighted_Price'] = param['Weighted_Price'].interpolate()

        param['Volume_(BTC)'] = param['Volume_(BTC)'].interpolate()
        param['Volume_(Currency)'] = param['Volume_(Currency)'].interpolate()
        param['High'] = param['High'].interpolate()
        param['Low'] = param['Low'].interpolate()
        return param

#ADF Test
def dickyfullertest(data):
    result=sm.tsa.stattools.adfuller(data)
    print('ADF-Statistics: {}'.format(result[0]))
    print('p-value : {}'.format(result[1]))
    if result[1]<0.05:
        print('Rejects the Null Hypothesis (H0) which signifies that the data is stationary.')
        return True
    else:
        print('Fail to reject the Null Hypothesis (H0) which signifies that the data has a unit root and is non-stationary.')
        return False

@app.route('/')
def index():
    return 'hai lulu'
 
@app.route('/upload', methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST' :
        #Uploaded File Flask
        uploaded_df = request.files['uploaded-file']
        # Extracting uploaded data file name
        data_filename = secure_filename(uploaded_df.filename)
        # flask upload file to database (defined uploaded folder in static path)
        uploaded_df.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))
        # Storing uploaded file path in flask session
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
        global df
        df = pd.read_csv(session['uploaded_data_file_path'])
        
        return 'Success'

@app.route('/read_data')
def read_data():
    # Retrieving uploaded file path from session
    data_file_path = session.get('uploaded_data_file_path', None)
    # read csv file in python flask (reading uploaded csv file from uploaded server location)
    uploaded_df = pd.read_csv(data_file_path)

@app.route('/show_data')
def showData():
    # Retrieving uploaded file path from session
    data_file_path = session.get('uploaded_data_file_path', None)
    # read csv file in python flask (reading uploaded csv file from uploaded server location)
    dataset = pd.read_csv(data_file_path)
    page = request.args.get('page', type=int, default=1)
    limit = request.args.get('limit', type=int, default=20)
    start = (page - 1) * limit
    end = page * limit
    data = dataset[start:end].fillna(value="NaN").to_dict('records')
    rows = data
    total_rows = len(dataset)
    total_page = math.ceil(total_rows/limit)
    resp = {
        'rows': rows,
        'header': list(dataset.columns),
        'total_pages': total_page,
        'total_rows': total_rows,
    }
    return resp

@app.route('/fill_missing')
def fillMissing():
    global df
    global filled_dataset
    global already_run
    if already_run == False:
        filled_dataset = fillMissingFunct(df)
    already_run = True
    page = request.args.get('page', type=int, default=1)
    limit = request.args.get('limit', type=int, default=20)
    start = (page - 1) * limit
    end = page * limit
    data = filled_dataset[start:end].to_dict('records')
    rows = data
    total_rows = len(filled_dataset)
    total_page = math.ceil(total_rows/limit)
    resp = {
        'rows': rows,
        'header': list(filled_dataset.columns),
        'total_pages': total_page,
        'total_rows': total_rows,
    }
    return resp

@app.route('/time_resampling')
def timeResampling():
    global df
    global already_run_tr
    global hourly_data
    global daily_data
    global time_resampled_data
    if already_run_tr == False:
        df = df.set_index('Timestamp')

        hourly_data = df.resample('1H').mean() #Hourly Resampling
        hourly_data = hourly_data.reset_index()

        daily_data = df.resample("24H").mean() #Daily resampling
        time_resampled_data = daily_data.reset_index()
    already_run_tr = True

    page = request.args.get('page', type=int, default=1)
    limit = request.args.get('limit', type=int, default=20)
    start = (page - 1) * limit
    end = page * limit
    data = time_resampled_data[start:end].to_dict('records')
    rows = data
    total_rows = len(time_resampled_data)
    total_page = math.ceil(total_rows/limit)
    resp = {
        'rows': rows,
        'header': list(time_resampled_data.columns),
        'total_pages': total_page,
        'total_rows': total_rows,
    }
    return resp

@app.route('/plot_data')
def plotData():
    global already_run_pd
    if already_run_pd == False: 
        # daily_data.reset_index(inplace=True)  
        # Plotting Data
        plt.figure(figsize=(18,10))
        plt.plot(daily_data['Weighted_Price'], label='Weighted Price')
        plt.title('Plot Bitcoin Price')
        plt.xlabel('Year')
        plt.ylabel('Bitcoin Price')
        plt.legend()
        plt.savefig('E:\Skripsi\Flask\hello_flask\Coba\staticFiles\plot_data.jpg')
        # fig = px.line(daily_data, x='Timestamp', y='Weighted_Price', title='Weighted Price with Range Slider and Selectors')
        # fig.update_layout(hovermode="x")  

        # fig.update_xaxes(
        #     rangeslider_visible=True,
        #     rangeselector=dict(
        #         buttons=list([
        #             dict(count=1, label="1m", step="month", stepmode="backward"),
        #             dict(count=6, label="6m", step="month", stepmode="backward"),
        #             dict(count=1, label="1y", step="year", stepmode="backward"),
        #             dict(count=2, label="2y", step="year", stepmode="backward"),
        #             dict(step="all") 
        #         ])
        #     )
        # )
    # fig.show()
    # plt.savefig('E:\Skripsi\Flask\hello_flask\Coba\staticFiles\plot_data.jpg')
    already_run_pd = True  
    resp = {
        'Grafik': url_for('static', filename='plot_data.jpg',  _external=True)
    }
    return resp

@app.route('/data_stationary')
def dataStationary():
    global daily_data
    global already_run_ds
    if already_run_ds == False:
        # daily_data = daily_data.reset_index(inplace=True)   
        daily_data['Open'] = daily_data['Open'].interpolate()
        daily_data['Close'] = daily_data['Close'].interpolate()
        daily_data['Weighted_Price'] = daily_data['Weighted_Price'].interpolate()

        daily_data['Volume_(BTC)'] = daily_data['Volume_(BTC)'].interpolate()
        daily_data['Volume_(Currency)'] = daily_data['Volume_(Currency)'].interpolate()
        daily_data['High'] = daily_data['High'].interpolate()
        daily_data['Low'] = daily_data['Low'].interpolate()

        already_run_ds = True
        #ADF Test
        adf_test = dickyfullertest(daily_data.Weighted_Price.dropna())
        if adf_test == True:
            resp1 = {
                    'Hasil ADF': dickyfullertest(daily_data.Weighted_Price.dropna())
                }
            return resp1
        # Differencng used if Fail
        else :
            adf_test_diff = dickyfullertest(daily_data.Weighted_Price.diff().dropna())
            resp2 = {
                    'Hasil ADF': dickyfullertest(daily_data.Weighted_Price.dropna()),
                    'Hasil Differencing ADF': dickyfullertest(daily_data.Weighted_Price.diff().dropna())
                }
            return resp2

@app.route('/feature_extraction')
def rollingWindows():
    global daily_data
    global already_run_fe
    if already_run_fe == False:
        daily_data.reset_index(drop=False, inplace=True)
        daily_data.set_index("Timestamp")
        lag_features = ["Open", "High", "Low", "Close", "Volume_(BTC)"]
        win1 = 3
        win2 = 7
        win3 = 30
        df_rolling3d = daily_data[lag_features].rolling(window=win1, min_periods=0)
        df_rolling7d = daily_data[lag_features].rolling(window=win2, min_periods=0)
        df_rolling30d = daily_data[lag_features].rolling(window=win3, min_periods=0)

        df_mean3d = df_rolling3d.mean().shift(1).reset_index()
        df_mean7d = df_rolling7d.mean().shift(1).reset_index()
        df_mean30d = df_rolling30d.mean().shift(1).reset_index()

        df_std3d = df_rolling3d.std().shift(1).reset_index()
        df_std7d = df_rolling7d.std().shift(1).reset_index()
        df_std30d = df_rolling30d.std().shift(1).reset_index()

        df_ema3d = daily_data[lag_features].ewm(span=3).mean()
        df_ema7d = daily_data[lag_features].ewm(span=7).mean()
        df_ema30d = daily_data[lag_features].ewm(span=30).mean()

        exp1 = daily_data[lag_features].ewm(span=12, adjust=False).mean()
        exp2 = daily_data[lag_features].ewm(span=26, adjust=False).mean()
        df_macd = exp1 - exp2
        df_signal = df_macd.ewm(span=9, adjust=False).mean()

        for feature in lag_features:
            daily_data["{0}_mean_lag{1}".format(feature,win1)] = df_mean3d[feature]
            daily_data["{0}_mean_lag{1}".format(feature,win2)] = df_mean7d[feature]
            daily_data["{0}_mean_lag{1}".format(feature,win3)] = df_mean30d[feature]
    
            daily_data["{0}_std_lag{1}".format(feature,win1)] = df_std3d[feature]
            daily_data["{0}_std_lag{1}".format(feature,win2)] = df_std7d[feature]
            daily_data["{0}_std_lag{1}".format(feature,win3)] = df_std30d[feature]
    
            daily_data["{0}_ewm{1}".format(feature,win1)] = df_ema3d[feature]
            daily_data["{0}_ewm{1}".format(feature,win2)] = df_ema7d[feature]
            daily_data["{0}_ewm{1}".format(feature,win3)] = df_ema30d[feature]
    
            daily_data['{0}_macd'.format(feature)]= df_macd[feature]
            daily_data['{0}_signal'.format(feature)]= df_signal[feature]

        daily_data.fillna(daily_data.mean(), inplace=True)

        daily_data.set_index("Timestamp", drop=False, inplace=True)
        daily_data["month"] = daily_data.Timestamp.dt.month
        daily_data["week"] = daily_data.Timestamp.dt.week
        daily_data["day"] = daily_data.Timestamp.dt.day
        daily_data["day_of_week"] = daily_data.Timestamp.dt.dayofweek

        already_run_fe = True

    page = request.args.get('page', type=int, default=1)
    limit = request.args.get('limit', type=int, default=20)
    start = (page - 1) * limit
    end = page * limit
    data = daily_data[start:end].to_dict('records')
    rows = data
    total_rows = len(data)
    total_page = math.ceil(total_rows/limit)
    resp = {
        'rows': rows,
        'header': list(daily_data.columns),
        'total_pages': total_page,
        'total_rows': total_rows,
    }
    return resp
    # print(daily_data)
    # return "Success"

@app.route('/arimax')
def Arimax():
    global daily_data
    global already_run_ar1
    global already_run_ar2
    if already_run_ar1 == False:
        daily_data['Timestamp'] = pd.to_datetime(daily_data['Timestamp'])
        already_run_ar1 = True
    df_total = daily_data[(daily_data['Timestamp'] > '2012') & (daily_data['Timestamp'] <= '2021')]
    df_train = daily_data[(daily_data['Timestamp'] >= '2012') & (daily_data['Timestamp'] <= '2020')]
    df_test = daily_data[(daily_data['Timestamp'] > '2020') & (daily_data['Timestamp'] <= '2021')]

    print('Total Shape :', df_total.shape)
    print('Train Shape :', df_train.shape)
    print('Test Shape :', df_test.shape)

    if already_run_ar2 == False:
        # Creating a list of exogenous or exemplary features.
        exogenous_features = ['Open_mean_lag3',
            'Open_mean_lag7', 'Open_mean_lag30', 'Open_std_lag3', 'Open_std_lag7',
            'Open_std_lag30', 'High_mean_lag3', 'High_mean_lag7', 'High_mean_lag30',
            'High_std_lag3', 'High_std_lag7', 'High_std_lag30', 'Low_mean_lag3',
            'Low_mean_lag7', 'Low_mean_lag30', 'Low_std_lag3', 'Low_std_lag7',
            'Low_std_lag30', 'Close_mean_lag3', 'Close_mean_lag7',
            'Close_mean_lag30', 'Close_std_lag3', 'Close_std_lag7',
            'Close_std_lag30', 'Volume_(BTC)_mean_lag3', 'Volume_(BTC)_mean_lag7',
            'Volume_(BTC)_mean_lag30', 'Volume_(BTC)_std_lag3',
            'Volume_(BTC)_std_lag7', 'Volume_(BTC)_std_lag30', 'Open_ewm3','Open_ewm7','Open_ewm30',
            'High_ewm3','High_ewm7','High_ewm30','Low_ewm3','Low_ewm7','Low_ewm30',
            'Close_ewm3', 'Close_ewm7', 'Close_ewm30',
            'Volume_(BTC)_ewm3', 'Volume_(BTC)_ewm7', 'Volume_(BTC)_ewm30', 
            'Open_macd', 'Close_macd', 'High_macd' , 'Low_macd', 'Volume_(BTC)_macd',
            'Open_signal', 'Close_signal', 'High_signal', 'Low_signal', 'Volume_(BTC)_signal',
            'month', 'week','day', 'day_of_week']
    
        # Leveraging Auto Arima to find the optimal parameters(p,d and q).
        model=pm.auto_arima(df_train.Weighted_Price, X = df_train[exogenous_features], trace=True, 
                        error_action="ignore", suppress_warnings=True)
        # Fitting the model based on train data.
        model.fit(df_train.Weighted_Price,  X = df_train[exogenous_features])
        already_run_ar2 = True
        
    # Predicting on the train data.
    df_train['ARIMAX forecast']=model.predict(n_periods=len(df_train), X = df_train[exogenous_features])

    # Solve Dataframe Chained Variables for Train Data
    hasil_prediksi = model.predict(n_periods=len(df_train), X = df_train[exogenous_features])
    hasil_prediksi = hasil_prediksi.reset_index(drop=True)

    df_train = df_train.reset_index(drop=True)
    df_train['ARIMAX forecast'] = hasil_prediksi
    df_train.set_index("Timestamp", drop=False, inplace=True)

    # Plotting the prediction vs train dataset values.
    plt.figure(figsize=(18,10))
    plt.plot(df_train['Weighted_Price'], label='Actual')
    plt.plot(df_train['ARIMAX forecast'],  label='Predicted')
    plt.title('Prediction on Train data')
    plt.xlabel('Timestamp')
    plt.ylabel('Bitcoin Price')
    plt.legend()
    plt.savefig('E:\Skripsi\Flask\hello_flask\Coba\staticFiles\train_plot_result.jpg')
    # plt.show()

    # Evaluating the prediciton Train Dataset using RMSE, MAE and R2 score.
    MAE_Train = mean_absolute_error(df_train['Weighted_Price'], df_train['ARIMAX forecast']).round(2)
    RMSE_Train = mean_squared_error(df_train['Weighted_Price'], df_train['ARIMAX forecast'], squared=False).round(2)

    # Predicting on the test data.
    df_test['ARIMAX forecast']=model.predict(n_periods=len(df_test), X = df_test[exogenous_features])

    # Solve Dataframe Chained Variables for Test Data
    hasil_prediksi_Test = model.predict(n_periods=len(df_test), X = df_test[exogenous_features])
    hasil_prediksi_Test = hasil_prediksi_Test.reset_index(drop=True)

    df_test = df_test.reset_index(drop=True)
    df_test['ARIMAX forecast'] = hasil_prediksi_Test
    df_test.set_index("Timestamp", drop=False, inplace=True)

    # Plotting the prediction vs test dataset values.
    plt.figure(figsize=(18,10))
    plt.plot(df_test['Weighted_Price'], label='Actual')
    plt.plot(df_test['ARIMAX forecast'], label='Predicted')
    plt.title('Prediction on Test data')
    plt.xlabel('Timestamp')
    plt.ylabel('Bitcoin Price')
    plt.legend()
    plt.savefig('E:\Skripsi\Flask\hello_flask\Coba\Grafik\test_plot_result.jpg')
    # plt.show()

    # Evaluating the prediciton Test Dataset using RMSE, MAE and R2 score.
    RMSE_Test = mean_squared_error(df_test['Weighted_Price'], df_test['ARIMAX forecast'], squared=False).round(2)
    MAE_Test = mean_absolute_error(df_test['Weighted_Price'], df_test['ARIMAX forecast']).round(2)
       
    resp = {
        "Grafik Train Data" : url_for('static', filename = 'train_plot_result.jpg',  _external=True),
        'RMSE Train': RMSE_Train,
        'MAE Train': MAE_Train,
        'Grafik Test Data': url_for('static', filename = 'test_plot_result.jpg',  _external=True),
        'RMSE Test' : RMSE_Test,
        'MAE Test' : MAE_Test
    }

    return resp

     # Check if operation was successful 
    if data_utama:
        # return a success response
        response = {
            'status_code': '200',
            'message': 'success get data'
        }
        return jsonify(response), 200
    else:
        # return an error response
        response = {
            'status': 'error',
            'message': 'Operation failed'
        }
 
if __name__=='__main__':
    app.run(debug = True)
