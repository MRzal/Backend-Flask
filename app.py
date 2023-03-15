import pandas as pd
from datetime import datetime
import os
import json
import math
import numpy as np
import seaborn as sns
from flask import Flask, render_template, request, session
from flask_paginate import Pagination, get_page_parameter
from flask_cors import CORS
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import plotly.express as px
from itertools import product
import warnings
import statsmodels.api as sm
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

df = '' 
already_run = False
    
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
    data = dataset[start:end].to_dict('records')
    rows = data
    total_rows = len(data)
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
    global already_run
    if already_run == False:
        df['Timestamp'] = [datetime.fromtimestamp(x) for x in df['Timestamp']] 
        df['Open'] = df['Open'].interpolate()
        df['Close'] = df['Close'].interpolate()
        df['Weighted_Price'] = df['Weighted_Price'].interpolate()

        df['Volume_(BTC)'] = df['Volume_(BTC)'].interpolate()
        df['Volume_(Currency)'] = df['Volume_(Currency)'].interpolate()
        df['High'] = df['High'].interpolate()
        df['Low'] = df['Low'].interpolate()
    already_run = True
    page = request.args.get('page', type=int, default=1)
    limit = request.args.get('limit', type=int, default=20)
    start = (page - 1) * limit
    end = page * limit
    data = df[start:end].to_dict('records')
    rows = data
    total_rows = len(data)
    total_page = math.ceil(total_rows/limit)
    resp = {
        'rows': rows,
        'header': list(df.columns),
        'total_pages': total_page,
        'total_rows': total_rows,
    }
    return resp

@app.route('/time_resampling')
@lru_cache(maxsize=None)
def timeResampling():
    global df
    df = df.set_index('Timestamp')

    hourly_data = df.resample('1H').mean()
    hourly_data = hourly_data.reset_index()
    print(hourly_data)

    return 'Success'

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
