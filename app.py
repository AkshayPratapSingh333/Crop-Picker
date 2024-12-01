from flask import Flask,request,render_template
import numpy
import pandas
import sklearn
import pickle

# Importing our model 
model  = pickle.load(open('crop_model.pkl','rb'))

# Creation of App for our Trained Model

app = Flask(__name__)

@app.route('/')

def index():
    return render_template("app.html")

@app.route("/predict",method=['POST'])
def predict():
    pass 





# Python MAIN
if __name__ == "__main__":
    app.run(debug=True)


