from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open('brst.pickle', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods = ['POST'])
def predict():

    input_features = [x for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name=['perc_premium_paid_by_cash_credit', 'Income',
       'application_underwriting_score', 'no_of_premiums_paid',
       'sourcing_channel', 'residence_area_type','age']
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)

    if output == 0:
        res_val = "** Premium h **"
    else:
        res_val = "no breast cancer"

    return render_template('home.html', prediction_text='Patient has {}'.format(res_val))


if __name__ == "__main__":
    app.run()