from flask import Flask,request,jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/",methods=['GET'])
def predict():
    return jsonify({"result":"successful deply"})

if __name__ == "__main__":
    app.run(debug=False,port=8080)


    