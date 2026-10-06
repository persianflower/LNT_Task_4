
import requests
import numpy as np
import cv2
import json
from PIL import Image
from flask import jsonify
from json import JSONEncoder

url = "http://127.0.0.1:8000/predict_json"


# creating a object
ankle_image = Image.open(r"/Users/ayesha/PycharmProjects/flaskapi/test_ankle.jpeg")
MAX_SIZE = (51, 73)

ankle_image.thumbnail(MAX_SIZE)

# creating thumbnail
ankle_image.save('ankle_thumb.png')

ankle = np.array(ankle_image)
ankle = cv2.resize(ankle, (28, 28))
ankle = np.dot(ankle[...,:3], [0.2989, 0.5870, 0.1140])
ankle = 255 - ankle
response = requests.post(url,json=ankle.tolist())
print(response)
print(response.json())

sandal_image = Image.open(r"/Users/ayesha/PycharmProjects/flaskapi/sandal.png")
MAX_SIZE = (51, 73)

sandal_image.thumbnail(MAX_SIZE)

# creating thumbnail
sandal_image.save('sandal_thumb.png')

sandal = np.array(sandal_image)
sandal = cv2.resize(sandal, (28, 28))
sandal = np.dot(sandal[...,:3], [0.2989, 0.5870, 0.1140])
sandal = 255 - sandal
response = requests.post(url,json=sandal.tolist())
print(response)
print(response.json())


shirt_image = Image.open(r"/Users/ayesha/PycharmProjects/flaskapi/shirt.jpeg")
MAX_SIZE = (51, 73)

shirt_image.thumbnail(MAX_SIZE)

# creating thumbnail
shirt_image.save('shirt_thumb.png')

shirt = np.array(shirt_image)
shirt = cv2.resize(shirt, (28, 28))
shirt = np.dot(shirt[...,:3], [0.2989, 0.5870, 0.1140])
shirt = 255 - shirt
response = requests.post(url,json=shirt.tolist())
print(response)
print(response.json())

