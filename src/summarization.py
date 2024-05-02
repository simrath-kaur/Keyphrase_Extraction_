import time
import json
from datetime import timedelta
from pandas import json_normalize
from extractive import ExtractiveSummarizer


from tqdm import tqdm
tqdm.pandas()


# ======================================================================================================================
# Define summarizer (transformersum)
# ======================================================================================================================

# using extractive model "distilroberta-base-ext-sum"
model = ExtractiveSummarizer.load_from_checkpoint("models\\epoch=3.ckpt")



# ======================================================================================================================
# NUS dataset
# ======================================================================================================================

# reading the initial JSON data using json.load()
file = 'datasets\\NUS.json'  # TEST data to evaluate the final model


# ======================================================================================================================
# Read data
# ======================================================================================================================

json_data = []
for line in open(file, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# convert json to dataframe
data = json_normalize(json_data)

print(data)



# count the running time of the program
start_time = time.time()


# ======================================================================================================================
# Summarize abstract and full-text (+ remove '\n')
# ======================================================================================================================

# extract abstract and full-text and create a summary
for index, abstract in enumerate(tqdm(data['abstract'])):
    # combine abstract + main body
    abstract_mainBody = abstract + ' ' + data['fulltext'][index]

    # remove '\n'
    abstract_mainBody = abstract_mainBody.replace('\n', ' ')

    # summarize abstract and full-text
    summarize_fulltext = model.predict(abstract_mainBody, num_summary_sentences=14)

    data['abstract'].iat[index] = summarize_fulltext

print(data)
print(data['abstract'][0])
print(data['abstract'][50])



total_time = str(timedelta(seconds=(time.time() - start_time)))
print("\n--- NUS %s running time ---" % total_time)


# ======================================================================================================================
# Save summarized NUS data to file
# ======================================================================================================================

summarized_file = 'datasets\\summarized_text\\NUS_summarized.csv'  # TEST data to evaluate the final model

data[['title', 'abstract', 'keywords']].to_csv(summarized_file, index=False)







