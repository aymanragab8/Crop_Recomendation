from flask import Flask, request, render_template
import numpy as np
import pickle

# Load your trained model
with open('CropRecomendation.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# The list of classes
classes = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute',
           'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange',
           'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon']

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Extract input values from the form
    n = float(request.form['n'])
    p = float(request.form['p'])
    k = float(request.form['k'])
    temperature = float(request.form['temperature'])
    ph = float(request.form['ph'])
    humidity = float(request.form['humidity'])

    # Create a numpy array from the inputs
    input_features = np.array([[n, p, k, temperature, ph, humidity]])

    # Predict the class and its probability
    prediction = model.predict(input_features)
    probabilities = model.predict_proba(input_features)

    predicted_class = classes[prediction[0]]
    predicted_probability = probabilities[0][prediction[0]]

    # Render the result in the template
    return render_template('index.html',
                           prediction_text=f'Predicted crop: {predicted_class} with probability: {predicted_probability:.2f}')


if __name__ == "__main__":
    app.run(debug=True)
