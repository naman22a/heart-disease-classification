from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('heart-disease.pkl', 'rb'))

@app.route('/api/predict',  methods=['POST'])
def predict_heart_disease():

    body = request.get_json()
    prediction = model.predict(np.array([body['fields']]))[0]

    if prediction:
        return jsonify({'disease': True})

    return jsonify({'disease': False})

if __name__ == '__main__':
    app.run(debug=True)
