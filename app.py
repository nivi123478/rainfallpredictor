from flask import Flask, render_template, request
from flask_cors import cross_origin
import pandas as pd
import pickle
import time

app = Flask(__name__, template_folder="template")

@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")

@app.route("/predict", methods=['POST'])
@cross_origin()
def predict():
    if request.method == "POST":
        start_time = time.time()  # Record the start time
        try:
            with open("best_xgb_classifier.pkl", "rb") as model_file:
                model = pickle.load(model_file)
            print("Model Loaded Successfully")
        except Exception as e:
            print("Error loading model:", e)
            return "An error occurred while loading the model."

        try:
            # Extracting data from the form
            rainfall = float(request.form['rainfall'])
            sunshine = float(request.form['sunshine'])
            windGustSpeed = float(request.form['windgustspeed'])
            humidity9am = float(request.form['humidity9am'])
            humidity3pm = float(request.form['humidity3pm'])
            pressure9am = float(request.form['pressure9am'])
            pressure3pm = float(request.form['pressure3pm'])
            cloud9am = float(request.form['cloud9am'])
            cloud3pm = float(request.form['cloud3pm'])
            rainToday = float(request.form['raintoday'])

            print("Input Data:", [rainfall, sunshine, windGustSpeed, humidity9am, humidity3pm, pressure9am, pressure3pm, cloud9am, cloud3pm, rainToday])

            # Create a DataFrame from the input data
            input_data = pd.DataFrame({
                'Rainfall': [rainfall],
                'Sunshine': [sunshine],
                'WindGustSpeed': [windGustSpeed],
                'Humidity9am': [humidity9am],
                'Humidity3pm': [humidity3pm],
                'Pressure9am': [pressure9am],
                'Pressure3pm': [pressure3pm],
                'Cloud9am': [cloud9am],
                'Cloud3pm': [cloud3pm],
                'RainToday': [rainToday]
            })

            print("Input DataFrame:")
            print(input_data)

            # Perform prediction
            try:
                pred = model.predict(input_data)
                print("Prediction Successful")
                output = pred[0]  # Assuming the output is a single prediction

                if output == 0:
                    return render_template("sunny.html")
                else:
                    return render_template("rainy.html")
            except Exception as e:
                print("Error during prediction:", e)
                return "An error occurred during prediction."
        except Exception as e:
            print("Error processing input data:", e)
            return "An error occurred while processing input data."
        finally:
            end_time = time.time()  # Record the end time
            execution_time = end_time - start_time  # Calculate execution time
            print(f"Execution time for /predict: {execution_time} seconds")

    # This part will execute if request.method != "POST"
    return render_template("home.html")

if __name__ == '__main__':
    app.run(debug=True)
