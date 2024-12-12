from flask import Flask, request, jsonify
import joblib
import pickle
import numpy as np
import os

# Load the trained model and vectorizer

model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(model_path, 'rb') as file:
    clf_loaded = pickle.load(file)


with open('vectorizer2.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Initialize the Flask application
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"


# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Process the text input
        custom_text = request.form.get('text', '').strip().lower()
        if not custom_text:
            return jsonify({'error': 'Input text is required'}), 400

        custom_text_vectorized = vectorizer.transform([custom_text])  # Wrap in a list
        prediction = clf_loaded.predict(custom_text_vectorized)
        prediction_proba = clf_loaded.predict_proba(custom_text_vectorized)

        # Prepare the response
        response = {
            'predicted_class': 'positive' if prediction[0] == 1 else 'negative',
            'positive_probability': round(prediction_proba[0][1], 4),
            'negative_probability': round(prediction_proba[0][0], 4)
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
