from flask import Flask, render_template, request
import numpy as np
import joblib

model = joblib.load('model/linear_regression.pkl')

app = Flask(__name__, template_folder='templates')

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = list(request.form.values())
    prediction = model.predict([features])
    output = round(prediction[0], 2)
    return render_template('main.html', prediction_text=f'The cab price is ${output}')

if __name__ == '__main__':
    app.run(debug=True)
