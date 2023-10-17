# Dependencies
from flask import Flask, request, jsonify, render_template
import joblib
import traceback
import pandas as pd
import numpy as np
import sys

# Your API definition
app = Flask(__name__)

# Load the model and model columns during initialization
lr = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        # Display the form
        return render_template('form.html')
    
    elif request.method == 'POST':
        if lr:
            if request.is_json:
                try:
                    json_ = request.get_json()
                    if isinstance(json_, dict):
                        data_df = pd.DataFrame([json_])
                        query = pd.get_dummies(data_df)
                        query = query.reindex(columns=model_columns, fill_value=0)

                        prediction = lr.predict(query)
                        prediction_text = "Survived" if prediction[0] == 1 else "Did Not Survive"

                        return jsonify({'prediction': prediction_text})

                    else:
                        return jsonify({"error": "Invalid JSON data format"})

                except:
                    return jsonify({'trace': traceback.format_exc()})
            else:
                if request.form:
                    age = float(request.form['age'])
                    sex = request.form['sex']
                    embarked = request.form['embarked']

                    data = pd.DataFrame({'Age': [age], 'Sex': [sex], 'Embarked': [embarked]})
                    data = pd.get_dummies(data)
                    data = data.reindex(columns=model_columns, fill_value=0)

                    prediction = lr.predict(data)
                    prediction_text = "Survived" if prediction[0] == 1 else "Did Not Survive"

                    return render_template('result.html', prediction=prediction_text)
                else:
                    return jsonify({"error": "Invalid request. Must be JSON data or form data."})

        else:
            print('Train the model first')
            return 'No model here to use'


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])  # This is for a command-line input
    except:
        port = 12345  # If you don't provide any port, the port will be set to 12345

    print('Model and model columns loaded')
    app.run(port=port, debug=True)
