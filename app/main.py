from flask import Flask , request, jsonify
from cnn_mnist import predict , prepare_image

app = Flask(__name__)

def checkFileExctention(filename:str):
    if filename.endswith(".png") or filename.endswith(".png") or filename.endswith(".png"):
        return True
    return False


@app.route("/")
def hello():
    return "helloword"


@app.route("/predict" , methods = ["POST"])
def predict_number():
    if request.method!="POST":
        return jsonify({"error" : "Unsupported Method"})
    file = request.files.get("image")
    if file==None or file.filename =="":
        return jsonify({"error": "file not Exist"})
    if not checkFileExctention(file.filename):
        return jsonify({"error" : "Unsupported Exctention"})
    
    try:
        img_bytes = file.read()
        image = prepare_image(image_bytes=img_bytes)
        result = predict(image)
        return jsonify({"status" : "success" , "result" : result})
    except:
        return jsonify({"error" : "500 server error"})