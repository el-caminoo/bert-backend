from flask import Flask,request,jsonify
from flask_cors import CORS
import torch
from transformers import BertTokenizer, BertForQuestionAnswering

app = Flask(__name__)
CORS(app)

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased')


def answer_question(question_text, document_text):
    input_ids = tokenizer.encode(question_text, document_text)

    # Search the input_ids for the first instance of the `[SEP]`
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    start_scores, end_scores = model(torch.tensor([input_ids]),token_type_ids=torch.tensor([segment_ids]))

    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespac
    for i in range(answer_start + 1, answer_end + 1):
    # If it's a subword token, then recombine it with the previous
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]
    return answer


@app.route("/",methods=['POST'])
def predict():
    question = request.data
    paragraph = request.data
    response = answer_question(question, paragraph)

    return jsonify({"result": response})

if __name__ == "__main__":
    app.run(debug=True, port=8000)


    

