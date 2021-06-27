from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pickle', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods = ['POST'])
def predict():

    input_features = [x for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name=['age','Income','sourcing_channel','residence_area_type','no_of_premiums_paid','application_underwriting_score','percentage_premium_paid_cash_credit','late_premium']
    df = pd.DataFrame(features_value, columns=features_name)
    prediction = model.predict(df)
    output = prediction[0]
    if output == 0:
       res_val ="Congrats... You don't have diabitic!"
    else:
        res_val ='You are diabitic!'
    return render_template('home.html', prediction_text=' {}'.format(res_val))


if __name__ == "__main__":
    app.run(debug=True)
