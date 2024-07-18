import json

from flask import Flask, request, jsonify
from flask_cors import CORS
from model_inference import Inference

app = Flask(__name__)
CORS(app)


@app.route('/process', methods=['POST'])
def process_input():
    # Get the input from the request
    input_json = request.get_json()
    input_json = json.dumps(input_json)
    input_data = json.loads(input_json)
    input_data = input_data['code']

    # Process the input using your Python code
    processed_output = process_input_data(input_data)

    # Return the processed output as a JSON response
    return jsonify({'markdown': processed_output})


def process_input_data(input_data):
    # Your Python code to process the input data
    inference = Inference()
    processed_output_string = inference.generate_output(input_data)
    # Return the processed output as a string
    return processed_output_string


if __name__ == '__main__':
    app.run()
