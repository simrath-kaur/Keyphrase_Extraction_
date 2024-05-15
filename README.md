For running this project, clone the github repo.

 Install dependencies specified in environment.

 Download the distilRoberta from transformerSum
library(refer the references), and place it under models\

 Download glove embedding (find the link in references),
and place it under GloVe\glove.6B directory.

 Then, go ahead and place the files in the src\ folder in root
folder.
 Then, you may run python app.py in the root folder.

 If you want to see all the NLP methods that we used,
checkout the Final_integration notebook

 If you want to run the files under experiments, first place
them in the root folder, and also place the datasets(find
link in references) and place them in
datasets\summarization_experiment.

References:
TransformerSum :
https://github.com/HHousen/TransformerSum
● DistilRoBERTa:https://huggingface.co/distilbert/distilroberta-base

● Datasets:

SemEval: https://github.com/boudinfl/akedatasets/tree/master/datasets/SemEval-2010
ACM: https://github.com/boudinfl/akedatasets/tree/master/datasets/ACM
NUS: https://paperswithcode.com/dataset/nus

● Pytorch-lightning Library:
https://lightning.ai/docs/pytorch/stable/

● NLTK: https://www.nltk.org/

● Glove
Embeddings:https://www.kaggle.com/datasets/daniel
willgeorge/glove6b100dtxt

● spaCy: https://pypi.org/project/spacy/

● Flask: https://flask.palletsprojects.com/en/3.0.x/
