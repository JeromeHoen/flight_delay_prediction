# -*- coding: utf-8 -*-
import pickle
import datetime

import pandas as pd    
import numpy as np
import scipy
import tensorflow as tf
from keras.models import load_model
import keras.backend as K
import holidays
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
application = app

carriers_dict = {
    'American Airlines (AA)': 'AA',
    'Alaska Airlines (AS)': 'AS',
    'JetBlue (B6)': 'B6',
    'Delta Air Lines (DL)': 'DL',
    'Atlantic Southeast Airlines (EV)': 'EV',
    'Frontier Airlines (F9)': 'F9',
    'Hawaiian Airlines (HA)': 'HA',
    'Spirit Airlines (NK)': 'NK',
    'SkyWest Airlines (OO)': 'OO',
    'Virgin America (UA)': 'UA',
    'United Airlines (VX)': 'VX',
    'Southwest Airlines (WN)': 'WN', 
}

dep_time_dict = {
    0: '0001-0559',
    1: '0001-0559',
    2: '0001-0559',
    3: '0001-0559',
    4: '0001-0559',
    5: '0001-0559',
    6: '0600-0659',
    7: '0700-0759',
    8: '0800-0859',
    9: '0900-0959',
    10: '1000-1059',
    11: '1100-1159',
    12: '1200-1259',
    13: '1300-1359',
    14: '1400-1459',
    15: '1500-1559',
    16: '1600-1659',
    17: '1700-1759',
    18: '1800-1859',
    19: '1900-1959',
    20: '2000-2059',
    21: '2100-2159',
    22: '2200-2259',
    23: '2300-2359'
}

bins = [-np.inf, 1, 21, 61, 121, 181, np.inf]

us_holidays = us_holidays = holidays.US()
X_cat_cols = ['CARRIER', 'ORIGIN', 'DEST', 'MONTH', 'DAY_OF_WEEK_H', 'DEP_TIME_BLK']


# open files
DIR = ""
#DIR = "/home/JeromeHoen/flight_delay_prediction/"

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

reg_model = load_model(DIR + "regresion_model.h5", custom_objects={'rmse': rmse})
classif_model = load_model(DIR + "classification_model.h5")

graph = tf.get_default_graph()

with open(DIR + "OneHotEncoder.pkl", 'rb') as f:
    OHE = pickle.load(f)
with open(DIR + "airports_dict.pkl", 'rb') as f:
    airports_dict = pickle.load(f)
airports_dist = pd.read_csv(DIR + "airports_distances.csv", index_col=[0, 1])
test_sample = pd.read_csv(DIR + "test_sample.csv")

inv_airports_dict = dict(zip(airports_dict.values(), airports_dict.keys()))
inv_carriers_dict = dict(zip(carriers_dict.values(), carriers_dict.keys()))

def fill_from_sample(df=test_sample):

    sample = df.sample(1)

    y = sample['ARR_DELAY'].values[0]

    return dict(
        carrier_name = inv_carriers_dict[sample['CARRIER'].values[0]],
        origin_name = inv_airports_dict[sample['ORIGIN'].values[0]],
        dest_name = inv_airports_dict[sample['DEST'].values[0]],
        dep_date = sample['FL_DATE'].values[0],
        dep_time = sample['DEP_TIME'].values[0],
        duration_hour = str(sample['DURATION_HOUR'].values[0]),
        duration_min = str(sample['DURATION_MIN'].values[0]),
        true_delay = delay_to_message(y)
    )

def get_results(
    carrier_name,
    origin_name,
    dest_name,
    dep_date,
    dep_time,
    duration_hour,
    duration_min
):

    X = transform_inputs(
        carrier_name,
        origin_name,
        dest_name,
        dep_date,
        dep_time,
        duration_hour,
        duration_min
    )
    
    with graph.as_default():
        reg = reg_model.predict(X)[0]
        classes_proba = class_predict(X)[0]
        
    results = dict(
        reg = delay_to_message(reg) + ".",
        classes = {},
    )
    for i, pred in enumerate(classes_proba):
        results['classes'][str(i)] = f"{pred * 100:.1f}%"

    return results

def delay_to_message(delay):
    delay = round(int(delay))
    if delay < 0:
        message = f"{abs(delay)} min in advance"
    elif delay == 0:
        message = "on time"
    elif delay > 0:
        message = f"{delay} min late"
    else:
        raise ValueError("Delay is not a number") 

    return message

def transform_date(date):
    date_time = datetime.datetime.strptime(date, '%Y-%m-%d')
    month = date_time.month
    if date_time in us_holidays:
        day_of_week_h = "H"
    else:
        day_of_week_h = date_time.isoweekday()
    return month, day_of_week_h

def transform_inputs(
    carrier_name,
    origin_name,
    dest_name,
    dep_date,
    dep_time,
    duration_hour,
    duration_min
):

    carrier_id = carriers_dict[carrier_name]
    origin_id = airports_dict[origin_name]
    dest_id = airports_dict[dest_name]

    dep_hour = int(dep_time[:2])
    dep_time_blk = dep_time_dict[dep_hour]

    distance = airports_dist.loc[(origin_id, dest_id)].values[0]
    duration = int(duration_hour) * 60 + int(duration_min)

    X = np.array([[distance, duration]])

    month, day_of_week_h = transform_date(dep_date)

    sample = np.array([[
        carrier_id,
        origin_id,
        dest_id,
        month,
        day_of_week_h,
        dep_time_blk
    ]], dtype=object)

    X = scipy.sparse.hstack((X, OHE.transform(sample)))
    X = scipy.sparse.csr_matrix(X)

    return X

def class_predict(X, outliers_rate=0.023):
    classif_no_outliers = classif_model.predict(X)
    classif_no_outliers = classif_no_outliers * (1 - outliers_rate)
    return np.hstack((classif_no_outliers, np.array([[outliers_rate]])))



@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        random = request.form.get('random')

        if random:
            results = fill_from_sample()
            return jsonify(results)

        else:
            inputs = request.form
            results = get_results(**inputs)
            return jsonify(results)   

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)