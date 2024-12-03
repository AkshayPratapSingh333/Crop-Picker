from flask import Flask, request, render_template
import numpy as np
import joblib
# Initialize Flask app
app = Flask(__name__)

# Load the model
ranfor = joblib.load('model.joblib')

# Crop dictionary with ID to name mapping
crop_dictionary = {
    1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut', 
    6: 'papaya', 7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon', 
    11: 'grapes', 12: 'mango', 13: 'banana', 14: 'pomegranate', 15: 'lentil', 
    16: 'blackgram', 17: 'mungbean', 18: 'mothbeans', 19: 'pigeonpeas', 
    20: 'kidneybeans', 21: 'chickpea', 22: 'coffee'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        # Extract form data
        N = int(request.form['Nitrogen'])
        P = int(request.form['Phosporus'])
        K = int(request.form['Potassium'])
        temper = float(request.form['Temperature'])
        hum = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rain = float(request.form['Rainfall'])

        # Prepare input for prediction
        crop_list = [N, P, K, temper, hum, ph, rain]
        crop_predict = np.array(crop_list).reshape(1, -1)
        
        # Make prediction
        predicted_crop = ranfor.predict(crop_predict)[0]  

        # Generate result
        if predicted_crop in crop_dictionary:
            result = "{} is a suitable crop for the environmental conditions you entered.".format(crop_dictionary[predicted_crop].capitalize())
        else:
            result = "Apologies, but we cannot recommend a crop based on the provided inputs."

    except Exception as e:
        # Handle unexpected errors
        result = f"An error occurred: {str(e)}"

    # Render the template with the result
    return render_template('index.html', result=result)

# Run the application
if __name__ == "__main__":
    app.run(debug=True)