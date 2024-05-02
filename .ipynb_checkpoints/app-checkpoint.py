from flask import Flask, request, jsonify
import json
from datetime import timedelta
from pandas import json_normalize
from extractive import ExtractiveSummarizer
import ast  
import pke
import string
import pandas as pd
import traditional_evaluation
import nltk
from nltk.corpus import stopwords
from pandas import json_normalize
from nltk.stem.snowball import SnowballStemmer as Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from tqdm import tqdm
import re

app = Flask(__name__)

# Load the ExtractiveSummarizer model
checkpoint_path = "models/epoch=3_modified.ckpt"
model = ExtractiveSummarizer.load_from_checkpoint(checkpoint_path)

# Function to write data to JSON file
def write_to_json(data):
    with open('datasets/input.json', 'w') as f:
        json.dump(data, f)

# Function to preprocess text
def preprocess_text(text):
    def get_contractions():
        contraction_dict = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because"}
        contractions_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
        return contraction_dict, contractions_re

    def replace_contractions(text):
        contractions, contractions_re = get_contractions()

        def replace(match):
            return contractions[match.group(0)]

        return contractions_re.sub(replace, text)

    newLine_tabs = '\t' + '\n'
    newLine_tabs_table = str.maketrans(newLine_tabs, ' ' * len(newLine_tabs))
    punctuation = string.punctuation  # + '\t' + '\n'
    # punctuation = punctuation.replace("'", '')  # do not delete '
    table = str.maketrans(punctuation, ' ' * len(punctuation))

    def remove_punct_and_non_ascii(text):
        clean_text = text.translate(table)
        clean_text = clean_text.encode("ascii", "ignore").decode()  # remove non-ascii characters
        # remove all single letter except from 'a' and 'A'
        clean_text = re.sub(r"\b[b-zB-Z]\b", "", clean_text)
        return clean_text

    def remove_brackets_and_contents(doc):
        ret = ''
        skip1c = 0
        for i in doc:
            if i == '[':
                skip1c += 1
            elif i == ']' and skip1c > 0:
                skip1c -= 1
            elif skip1c == 0:
                ret += i
        return ret

    def remove_newline_tabs(text):
        return text.replace('\n', ' ').replace('\t', ' ')

    def remove_references(doc):
        clear_doc = doc.translate(newLine_tabs_table)
        clear_doc = re.sub(r'[A-Z][a-z]+,\s[A-Z][a-z]*\. et al.,\s\d{4}', "REFPUBL", clear_doc)
        clear_doc = re.sub("[A-Z][a-z]+ et al. [0-9]{4}", "REFPUBL", clear_doc)
        clear_doc = re.sub("[A-Z][a-z]+ et al.", "REFPUBL", clear_doc)
        return clear_doc

    text = replace_contractions(text)
    text = remove_punct_and_non_ascii(text)
    text = remove_brackets_and_contents(text)
    text = remove_newline_tabs(text)
    text = remove_references(text)
    return text

# Function to extract keyphrases
def extract_keyphrases(text):
    def load_glove_model(glove_file):
        word_embeddings = {}
        with open(glove_file, encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                word_embeddings[word] = coefs
        return word_embeddings

    def phrase_identification_tfidf(data):
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(data['abstract'])
        return tfidf_matrix

    def load_bert_embeddings(data):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        embeddings = []
        for abstract in data['abstract']:
            input_ids = tokenizer.encode(abstract, add_special_tokens=True, max_length=512, truncation=True)
            input_ids = torch.tensor(input_ids).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_ids)
                embeddings.append(outputs[0][:, 0, :].numpy())  # Extracting the [CLS] token embedding
        return embeddings

    def extract_keyphrases(data, tfidf_matrix, glove_model):
        gold_keyphrases = []  
        pred_keyphrases = []  
        for indx, abstract_document in enumerate(data['abstract']):
            abstract_document = preprocess_text(abstract_document)

            gold_keyphrases.append([[Stemmer('porter').stem(keyword) for keyword in keyphrase.split()] for keyphrase in data['keyword'][indx].split(';')])

            extractor = pke.unsupervised.MultipartiteRank()
            extractor.load_document(input=abstract_document, normalization="stemming")
            pos = {'NOUN', 'PROPN', 'ADJ'}
            stoplist = list(string.punctuation)
            stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
            stoplist += stopwords.words('english')
            extractor.candidate_selection(pos=pos)

            # Compute similarity scores between candidate phrases and document using GloVe embeddings
            candidate_scores = []
            for phrase in extractor.candidates:
                candidate_embedding = np.mean([glove_model.get(word, np.zeros((100,))) for word in phrase[0].split()], axis=0)
                document_embedding = np.mean([glove_model.get(word, np.zeros((100,))) for word in abstract_document.split()], axis=0)
                similarity_score = np.dot(candidate_embedding, document_embedding) / (np.linalg.norm(candidate_embedding) * np.linalg.norm(document_embedding))
                candidate_scores.append(similarity_score)  # Keep similarity scores directly

            # Using similarity scores for candidate phrases weighting
            extractor.candidate_weighting(method='average')  # Set method only
            pred_kps = extractor.get_n_best(n=10)
            pred_keyphrases.append([kp[0].split() for kp in pred_kps])
        return pred_keyphrases, gold_keyphrases

    data = pd.DataFrame({
        'abstract': [text],
        'keyword': ['']  # Dummy keyword column for compatibility
    })

    glove_model = load_glove_model('GloVe/glove.6B/glove.6B.100d.txt')
    tfidf_matrix = phrase_identification_tfidf(data)
    pred_keyphrases, _ = extract_keyphrases(data, tfidf_matrix, glove_model)
    return pred_keyphrases[0]

# Route to receive user input and update JSON file
@app.route('/update', methods=['POST'])
def update_json():
    data = request.get_json()
    # Preprocess the abstract text
    data['abstract'] = preprocess_text(data['abstract'])
    # Write data to JSON file
    write_to_json(data)
    return jsonify({'message': 'JSON file updated successfully'})

# Route to get predicted keyphrases
@app.route('/predict', methods=['GET'])
def predict_keyphrases():
    # Read data from JSON file
    with open('datasets/input.json', 'r') as f:
        data = json.load(f)
    # Extract keyphrases
    pred_keyphrases = extract_keyphrases(data['abstract'])
    return jsonify({'predicted_keyphrases': pred_keyphrases})

if __name__ == '__main__':
    app.run(debug=True)
