# Implementation of Prediction API using Flask for CNN Network Trained on MNIST Dataset
This project is an implementation of a prediction API using Flask for a Convolutional Neural Network (CNN) that is trained on the MNIST dataset. The MNIST dataset consists of handwritten digits and is a popular benchmark dataset for image classification tasks. The CNN is trained on this dataset to recognize digits from 0 to 9.


## Dataset

The MNIST dataset consists of 60,000 training images and 10,000 test images. The images are grayscale and have a resolution of 28 x 28 pixels.


## Requirements

Python 3.6 or later
Flask
NumPy
TensorFlow
Keras


## Usage

start the prediction API server, run the following command
flask run
note: first you should change FLASK_APP variable to main.py
This will start the Flask server on http://localhost:5000. You can then use a tool like Postman or cURL to send POST requests to the server with a JSON payload that contains a base64-encoded image of a handwritten digit. The server will then return a JSON response with the predicted digit and the probability of the prediction.
You can use test/test.py for test server
