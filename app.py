from flask import Flask, render_template, request, jsonify
import json
from summarize_text import summarize_text  # Import your summarization script function

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    
    # Save the JSON data to input.json
    with open('input.json', 'w') as f:
        json.dump(data, f)
    
    # Call the summarize_text function with the file path
    predicted_keyphrases = summarize_text('input.json')
    
    return jsonify({'predicted_keyphrases': predicted_keyphrases})


if __name__ == '__main__':
    app.run(debug=True)
