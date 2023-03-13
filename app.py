from flask import Flask, render_template, request, session
import pandas as pd
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
import math
 
#*** Flask configuration
 
# Define folder to save uploaded files to process further
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
 
# Define allowed files (for this example I want only csv file)
ALLOWED_EXTENSIONS = {'csv'}
 
app = Flask(__name__, template_folder='templateFiles', static_folder='staticFiles')
# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = '/home/biki33/projek/Backend-Flask/dataset'
CORS(app)
 
# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'
 
 
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
        
        return 'Success'

@app.route('/read_data')
def read_data():
    # Retrieving uploaded file path from session
    data_file_path = session.get('uploaded_data_file_path', None)
    # read csv file in python flask (reading uploaded csv file from uploaded server location)
    uploaded_df = pd.read_csv(data_file_path)

@app.route('/pagination')
def pagination():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    data = uploaded_df
    start = (page - 1) * per_page
    end = start + per_page
    rows = data[start:end]
    total_rows = len(data)
    total_pages = (total_rows // per_page) + (1 if total_rows % per_page > 0 else 0)
    #return render_template('index.html', rows=rows, page=page, per_page=per_page, total_rows=total_rows, total_pages=total_pages)

@app.route('/show_data')
def showData():
    # Retrieving uploaded file path from session
    data_file_path = session.get('uploaded_data_file_path', None)
    # read csv file in python flask (reading uploaded csv file from uploaded server location)
    dataset = pd.read_csv(data_file_path)
    data = dataset.to_dict('records')
    page = request.args.get('page', type=int, default=1)
    limit = request.args.get('limit', type=int, default=20)
    start = (page - 1) * limit
    end = page * limit
    rows = data[start:end]
    total_rows = len(data)
    total_page = math.ceil(total_rows/limit)
    resp = {
        'rows': rows,
        'header': list(dataset.columns),
        'total_pages': total_page,
        'total_rows': total_rows,
    }
    return resp
 
if __name__=='__main__':
    app.run(debug = True)