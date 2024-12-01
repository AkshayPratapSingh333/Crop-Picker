from flask import Flask, render_template, request
import numpy as np
import pickle

# Initialize Flask app
app = Flask(__name__)

# importing model
model = pickle.load(open('model.pkl','rb'))


# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)


    predicted_crop = model.predict(final_features)

    crop_dictionary = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}



        # Generate result
    if predicted_crop in crop_dictionary:
            result = f"{predicted_crop.capitalize()} is a suitable crop for the environmental conditions you entered."
    else:
            result = "Apologies, but we cannot recommend a crop based on the provided inputs."


    # Render the template with the result
    return render_template('app.html', result=result)





# Run the application
if __name__ == "__main__":
    app.run(debug=True)
