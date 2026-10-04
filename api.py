
import requests
import numpy as np
import cv2
import json
from PIL import Image
from flask import jsonify
from json import JSONEncoder

url = "http://127.0.0.1:8000/predict"

# creating a object
image = Image.open(r"/Users/ayesha/PycharmProjects/flaskapi/test_ankle.jpeg")
MAX_SIZE = (51, 73)

image.thumbnail(MAX_SIZE)

# creating thumbnail
image.save('ankle_thumb.png')

ankle = np.array(image)
ankle = cv2.resize(ankle, (28, 28))
ankle = np.dot(ankle[...,:3], [0.2989, 0.5870, 0.1140])
ankle = 255 - ankle
response = requests.post(url,json=ankle.tolist())
print(response)
print(response.json())

