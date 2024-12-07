from flask import Flask, request, render_template
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the machine learning model and scaler
try:
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model or scaler: {e}")

# Initialize Flask app
app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    """
    Renders the main page with the input form.
    """
    return render_template('index.html', result='')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles crop prediction based on user inputs and recommends the best crop.
    """
    # Extract form data
    nitrogen = request.form['Nitrogen']
    phosphorus = request.form['Phosphorus']
    potassium = request.form['Potassium']
    temperature = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    # Ensure all values are provided
    if not all([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]):
        result = "Missing input values. Please fill all fields."
        return render_template('index.html', result=result)

    # Prepare input data for the model
    input_features = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
    one_prediction = np.array(input_features).reshape(1, -1)

    # Debugging: Check input shape
    print("Shape of input for prediction:", one_prediction.shape)

    try:
        # Apply MinMaxScaler and StandardScaler
        scalar_features = ms.transform(one_prediction)  # Apply MinMaxScaler
        print("Input after MinMaxScaler transformation:", scalar_features)

        final_features = scaler.transform(scalar_features)  # Apply StandardScaler
        print("Input after StandardScaler transformation:", final_features)

        # Predict the crop
        prediction = model.predict(final_features)
        print("Model prediction:", prediction)

    except Exception as e:
        result = f"Error during prediction: {e}"
        return render_template('index.html', result=result)

    # Crop dictionary with mappings
    crop_dictionary = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
        8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }

    # Check prediction and map to crop
    if prediction[0] in crop_dictionary:
        crop = crop_dictionary[prediction[0]]
        result = f"{crop} is a suitable crop for the environmental conditions you entered."
    else:
        result = "Apologies, but we cannot recommend a crop based on the provided inputs."

    # Render the result
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
