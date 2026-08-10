import os
from flask import Flask,render_template,request
import pandas as pd
import joblib

app = Flask(__name__)

# Load saved model and encoder
model = joblib.load("delivery_time_model.pkl")
lb = joblib.load("traffic_encoder.pkl")
columns=joblib.load("model_columns.pkl")
scaler=joblib.load("delivery_sacler.pkl")

@app.route('/')
def index():
    return render_template("index.html")
  
@app.route('/about') 
def about():
    return render_template("about.html")

@app.route('/predict',methods=['GET','POST']) 
def predict():
    prediction_text = None
    
    if request.method == 'POST':
        #take user input
        distance = float(request.form['distance'])
        t_level = request.form['t_level']
        p_time = int(request.form['p_time'])
        experience = float(request.form['experience'])
        weather = request.form['weather']
        time = request.form['time']
        vehicle = request.form['vehicle']
        
        # Create dictionary with all columns = 0
        input_dict = {col: 0 for col in columns}

        # Fill numeric values
        input_dict['Distance_km'] = distance
        input_dict['Preparation_Time_min'] = p_time
        input_dict['Courier_Experience_yrs'] = experience

        # Encode Traffic Level
        input_dict['Traffic_Level'] = lb.transform([t_level])[0]

        # Dummy encoding manually
        # Weather
        input_dict[f"Weather_{weather}"] = 1

        # Time of Day
        input_dict[f"Time_of_Day_{time}"] = 1

        # Vehicle Type
        input_dict[f"Vehicle_Type_{vehicle}"] = 1

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Ensure correct column order
        input_df = input_df[columns]

        #Apply scaler
        input_df=scaler.transform(input_df)

        pred=model.predict(input_df)[0]
        
        prediction_text = f"Estimated Delivery Time: {round(pred, 2)} minutes"
    
    return render_template('predict.html',prediction_text=prediction_text)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, port=port, host='0.0.0.0')