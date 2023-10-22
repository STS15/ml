# model_server.py
from flask import Flask, request, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)

# Load model and tokenizer
model_identifier = "MODEL_IDENTIFIER"  # Replace with your actual environment variable or direct model name
model = AutoModelForSeq2SeqLM.from_pretrained(model_identifier)
tokenizer = AutoTokenizer.from_pretrained(model_identifier)

@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.json
    inputs = json_data["inputs"]
    
    # Encode and send to model
    inputs = tokenizer.encode(inputs, return_tensors="pt")
    out = model.generate(inputs)

    # Decode and send back
    decoded_output = [tokenizer.decode(t, skip_special_tokens=True) for t in out]
    return jsonify(decoded_output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
