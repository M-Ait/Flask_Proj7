import pickle
from flask import Flask, jsonify, render_template, request
import pandas as pd
from waitress import serve

app = Flask(__name__)

with open('static/LGBM_final.pkl', 'rb') as file:
    lgbm = pickle.load(file)

app_test = pd.read_csv('static/app_test.csv')
client_ID = list(app_test.SK_ID_CURR)

@app.route('/') # / = No endpoint : local url
def index():
    return render_template('index.html')

@app.route('/possible_input/')
def possible_input():
    return jsonify({"model": "LGBM optimise",
                    "possible_client_ID" : client_ID})


@app.route('/predict/<int:sk_id>')
def predict_get(sk_id):
    if sk_id in client_ID:
        client_info = app_test[app_test['SK_ID_CURR']==sk_id].drop(columns=['SK_ID_CURR'])
        predict_init = lgbm.predict(client_info)[0]
        predict_proba = round(lgbm.predict_proba(client_info)[0][1], 2)
        if predict_proba > 0.1:
            predict = 1
        else:
            predict = 0
    else:
        predict_init = predict_proba = predict = "client inconnu"
    return jsonify({'client_ID' : str(sk_id),
                    'prediction_revisitee' : str(predict),
                    'prediction_initiale': str(predict_init),
                    'probabilite_refus': str(predict_proba)})

@app.route('/predict/', methods=['GET', 'POST'])
def predict_get1():
    sk_id = request.form.get('sk_id', type=int)
    if sk_id in client_ID:
        client_info = app_test[app_test['SK_ID_CURR']==sk_id].drop(columns=['SK_ID_CURR'])
        predict_init = lgbm.predict(client_info)[0]
        predict_proba = round(lgbm.predict_proba(client_info)[0][1], 2)
        if predict_proba > 0.1:
            predict = 1
        else:
            predict = 0
    else:
        predict_init = predict_proba = predict = "client inconnu"
    return jsonify({'client_ID' : str(sk_id),
                    'prediction_revisitee' : str(predict),
                    'prediction_initiale': str(predict_init),
                    'probabilite_refus': str(predict_proba)})


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8080)
