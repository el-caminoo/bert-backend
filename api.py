from flask import Flask,request,jsonify
from flask_cors import CORS
import torch
from transformers import BertTokenizer, BertModel, BertForQuestionAnswering

app = Flask(__name__)
CORS(app)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

@app.route("/",methods=['POST'])
def predict():
    return jsonify({"result":output})

if __name__ == "__main__":
    app.run(debug=True, port=8000)


    

