import time
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
tqdm.pandas()
import torch
    
# Load the checkpoint file with map_location=torch.device('cpu')
checkpoint = torch.load("models\\epoch=3.ckpt", map_location=torch.device('cpu'))

# Rename the key from 'pytorch-ligthning_version' to 'pytorch-lightning_version'
checkpoint['pytorch-lightning_version'] = checkpoint.pop('pytorch-ligthning_version')

# Save the modified checkpoint
torch.save(checkpoint, "models\\epoch=3_modified.ckpt")

# Now load the model using ExtractiveSummarizer.load_from_checkpoint
from extractive import ExtractiveSummarizer

# Load the model checkpoint
checkpoint_path = "models\\epoch=3_modified.ckpt"
model = ExtractiveSummarizer.load_from_checkpoint(checkpoint_path)

def summarize_text(input):
    
    file=input
    json_data = []
    for line in open(file, 'r', encoding="utf8"):
        json_data.append(json.loads(line))
    
    # convert json to dataframe
    data = json_normalize(json_data)
    
    print(data)
    for index, abstract in enumerate(tqdm(data['abstract'])):
        # combine abstract + main body
        abstract_mainBody = abstract + ' ' + data['fulltext'][index]
    
        # remove '\n'
        abstract_mainBody = abstract_mainBody.replace('\n', ' ')
    
        # summarize abstract and full-text
        summarize_fulltext = model.predict(abstract_mainBody, num_summary_sentences=14)
    
        data['abstract'].iat[index] = summarize_fulltext
    file_abstract = input 
    data_summaries = data[['title', 'abstract']]
    # Read data
    json_data = []
    for line in open(file_abstract, 'r', encoding="utf8"):
        json_data.append(json.loads(line))
    
    data_abstract = json_normalize(json_data)
    # Preprocessing Functions
    import re
    
    def get_contractions():
        contraction_dict = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                            "could've": "could have", "couldn't": "could not", "didn't": "did not",
                            "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
                            "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
                            "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                            "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
                            "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",
                            "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have",
                            "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
                            "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
                            "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                            "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                            "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                            "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                            "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                            "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                            "she'll've": "she will have", "she's": "she is", "should've": "should have",
                            "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                            "so's": "so as", "this's": "this is", "that'd": "that would",
                            "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                            "there'd've": "there would have", "there's": "there is", "here's": "here is",
                            "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                            "they'll've": "they will have", "they're": "they are", "they've": "they have",
                            "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
                            "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",
                            "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
                            "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is",
                            "when've": "when have", "where'd": "where did", "where's": "where is",
                            "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                            "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
                            "will've": "will have", "won't": "will not", "won't've": "will not have",
                            "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                            "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",
                            "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
                            "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                            "you're": "you are", "you've": "you have", "nor": "not", "'s": "s", "s'": "s"}
    
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
    #punctuation = punctuation.replace("'", '')  # do not delete '
    table = str.maketrans(punctuation, ' '*len(punctuation))
    
    def remove_punct_and_non_ascii(text):
        clean_text = text.translate(table)
        clean_text = clean_text.encode("ascii", "ignore").decode()  # remove non-ascii characters
        # remove all single letter except from 'a' and 'A'
        clean_text = re.sub(r"\b[b-zB-Z]\b", "", clean_text)
        return clean_text
    def remove_brackets_and_contents(doc):
        """
        remove parenthesis, brackets and their contents
        :param doc: initial text document
        :return: text document without parenthesis, brackets and their contents
        """
        ret = ''
        skip1c = 0
        # skip2c = 0
        for i in doc:
            if i == '[':
                skip1c += 1
            # elif i == '(':
            # skip2c += 1
            elif i == ']' and skip1c > 0:
                skip1c -= 1
            # elif i == ')'and skip2c > 0:
            # skip2c -= 1
            elif skip1c == 0:  # and skip2c == 0:
                ret += i
        return ret
    
    def remove_newline_tabs(text):
        return text.replace('\n', ' ').replace('\t', ' ')
    
    def remove_references(doc):
        """
        remove references of publications (in document text)
        :param doc: initial text document
        :return: text document without references
        """
        # delete newline and tab characters
        clear_doc = doc.translate(newLine_tabs_table)
    
        # remove all references of type "Author, J. et al., 2014"
        clear_doc = re.sub(r'[A-Z][a-z]+,\s[A-Z][a-z]*\. et al.,\s\d{4}', "REFPUBL", clear_doc)
    
        # remove all references of type "Author et al. 1990"
        clear_doc = re.sub("[A-Z][a-z]+ et al. [0-9]{4}", "REFPUBL", clear_doc)
    
        # remove all references of type "Author et al."
        clear_doc = re.sub("[A-Z][a-z]+ et al.", "REFPUBL", clear_doc)
    
        return clear_doc
    
    def preprocessing(text):
        text = replace_contractions(text)
        text = remove_punct_and_non_ascii(text)
        text = remove_brackets_and_contents(text)
        text = remove_newline_tabs(text)
        text = remove_references(text)
        return text
    # NLP Components
    def extract_keyphrases(data, tfidf_matrix, glove_model): 
        pred_keyphrases = []  
        for indx, abstract_document in enumerate(data['abstract']):
            abstract_document = preprocessing(abstract_document)
            
            
            
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
        return pred_keyphrases
    def phrase_identification_tfidf(data):
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(data['abstract'])
        return tfidf_matrix
    
    def load_glove_model(glove_file):
        word_embeddings = {}
        with open(glove_file, encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                word_embeddings[word] = coefs
        return word_embeddings
    
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
    import re
    def combine_text(data):
        for index, abstract in enumerate(data['abstract']):
            title_abstract_summary = data['title'][index] + '. ' + abstract
            title_abstract_summary = preprocessing(title_abstract_summary)
            data['abstract'].iat[index] = title_abstract_summary
        if 'keywords' in data.columns:
            data.rename(columns={"keywords": "keyword"}, inplace=True)
        return data
    
    data_abstract = combine_text(data_abstract)
    data_summaries = combine_text(data_summaries)
    # Extract keyphrases
    tfidf_matrix = phrase_identification_tfidf(data_abstract)
    glove_model = load_glove_model('GloVe\\glove.6B\\glove.6B.100d.txt')
    pred_keyphrases_abstract = extract_keyphrases(data_abstract, tfidf_matrix, glove_model)
    pred_keyphrases_sum = extract_keyphrases(data_summaries, tfidf_matrix, glove_model)
    # Combine the lists
    combined_lists = pred_keyphrases_sum[0] + pred_keyphrases_abstract[0]


    unique_sets = {tuple(lst) for lst in combined_lists}


    unique_lists = [list(lst) for lst in unique_sets]
    return (unique_lists)
