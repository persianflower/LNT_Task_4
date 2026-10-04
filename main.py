
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from tensorflow import keras, float16

app = Flask(__name__)

with open('fashion_model.pkl','rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Fashion data prediction app working!"

@app.route('/predict',methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error":"No data provided"}),400
        data = np.array(data, dtype=np.float32)
        if data.size != 28 * 28:
            return jsonify({
                "error": f"Expected 784 pixels, got {data.size}"
            }), 400
        data = data.reshape(1, 28, 28, 1)
        data = data / 255.0
        prediction = model.predict(data)
        predicted_class = int(np.argmax(prediction[0]))
        return jsonify({"prediction": predicted_class})
    except Exception as e:
        return jsonify({"error":str(e)}),400

if __name__=="__main__":
    app.run(debug=True,port=8000)
'''

import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained pickle model
with open("fashion_model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return "Fashion data prediction app working!"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Receive JSON data
        data = request.get_json()

        if data is None:
            return jsonify({"error": "No JSON data provided"}), 400

        # Convert JSON list to NumPy array
        data = np.array(data, dtype=np.float32)

        # Expected input: 28 x 28 x 1
        if data.size != 28 * 28:
            return jsonify({
                "error": f"Expected 784 pixels, got {data.size}"
            }), 400

        # Convert to: (1, 28, 28, 1)
        data = data.reshape(1, 28, 28, 1)

        # Scale pixels from 0-255 to 0-1
        data = data / 255.0

        # Make prediction
        prediction = model.predict(data)

        # Get predicted class
        predicted_class = int(np.argmax(prediction[0]))

        # Get confidence
        confidence = float(np.max(prediction[0]))

        class_names = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot"
        ]

        return jsonify({
            "prediction": predicted_class,
            "class_name": class_names[predicted_class],
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=8000)

'''
