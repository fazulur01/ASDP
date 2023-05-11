from flask import Flask, render_template, request
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import plotly.express as px
import plotly.io as pio
import plotly.utils

app = Flask(__name__)

#loading the model
model = pickle.load(open('savedmodel.sav', 'rb'))

@app.route('/')
def home():
    result = ''
    return render_template('index_1.html', **locals())


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    country = float(request.form['Country'])
    ISO = float(request.form['ISO'])
    sex = float(request.form['Sex'])
    year = float(request.form['Year'])
    ASDP = float(request.form['ASDP'])
    lower_uncertainty = float(request.form['Lower_95_uncertainty'])
    result = model.predict([[country, ISO, sex, year, ASDP, lower_uncertainty]])[0]
    return render_template('index.html', **locals())

@app.route('/chart1')
def chart1():
    # Assuming you have a DataFrame or other suitable data structure for the chart data
    fig = px.bar(model, x='Sex', y='Age-standardised diabetes prevalence', title='Average age-standardized death rate (ASDP) by sex')

    # Convert plot to JSON format using plotly.io.to_json()
    fig_json = pio.to_json(fig)

    # Return JSON chart and metadata
    header = "Comparison of average ASDP by sex"
    description = """
    This chart compares the average age-standardized death rate (ASDP) between males and females.
    """
    return render_template('index_2.html', graphJSON=fig_json, header=header, description=description)


if __name__ == '__main__':
    app.run(debug=True)
    