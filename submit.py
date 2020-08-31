#print ('Team TalkIsCheap sends its Hello World')


import json
import os
import pandas as pd
from random import randint
import numpy as np
import re
import spacy
import string
import pickle
from spacy.lang.en.stop_words import STOP_WORDS

#if __name__ == "__main__":
# These are the file paths where the validation/test set will be mounted (read only)
# into your Docker container.
METADATA_FILEPATH = '/usr/local/dataset/metadata.json'
ARTICLES_FILEPATH = '/usr/local/dataset/articles'

# This is the filepath where the predictions should be written to.
PREDICTIONS_FILEPATH = '/usr/local/predictions.txt'

# Read in the metadata file.
with open(METADATA_FILEPATH, 'r') as f:
    claims = json.load(f)

# Inspect the first claim.
claim = claims[0]
print('Claim:', claim['claim'])
print('Speaker:', claim['claimant'])
print('Date:', claim['date'])
print('Related Article Ids:', claim['related_articles'])

# Print the first evidence article.
idx = claim['related_articles'][0]
print('First evidence article id:', idx)
# with open(os.path.join(ARTICLES_FILEPATH, '%d.txt' % idx), 'r') as f:
#     print(f.read())

# Create a predictions file.

# Load model and feature preprocess 
model = pickle.load(open('model_nov4.sav', 'rb'))
pre_process = pickle.load(open('pre_process_nov4.sav', 'rb'))

df = pd.DataFrame(claims)

#add has author 
df["author"]= np.where(df["claimant"]=="",0,1)
#replace missing with others first
df["claimant"] = np.where(df["claimant"]=="","other",df["claimant"])

data = df.copy()
def normalize(text):
    """
    Given a text, cleans and normalizes it. Feel free to add your own stuff.
    """
    text = text.lower()
    # Replace ips
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', text)
    # Isolate punctuation
    text = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', text)
    # Remove some special characters
    text = re.sub(r'([\;\:\|•«\n])', ' ', text)
    # Remove some special characters
    text = re.sub(r'([\" "\"''"\が\"' '"\、\❂])', ' ', text)
    # Replace numbers and symbols with language
    text = text.replace('&', ' and ')
    text = text.replace('@', ' at ')
    text = text.replace('-', '')
    text = text.replace('“', '')
    text = text.replace("'", '')  
       
    #text = text.replace('0', ' zero ')
    #text = text.replace('1', ' one ')
    #text = text.replace('2', ' two ')
    #text = text.replace('3', ' three ')
    #text = text.replace('4', ' four ')
    #text = text.replace('5', ' five ')
    #text = text.replace('6', ' six ')
    #text = text.replace('7', ' seven ')
    #text = text.replace('8', ' eight ')
    #text = text.replace('9', ' nine ')
    return text

#Remove punctuation of each observation
data['claim'] = data['claim'].apply(normalize)

#function to remove punctuation
def remove_punctuation(text):
    '''a function for removing punctuation'''
    import string
    # replacing the punctuations with no space, 
    # which in effect deletes the punctuation marks 
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped of punctuation marks
    return text.translate(translator)
data['claim'] = data['claim'].apply(remove_punctuation)

nlp = spacy.load('en_core_web_sm')
nlp.Defaults.stop_words |= {"s","t","el","…","f'","u'","1'","ve","u"}

# Clean text before feeding it to spaCy
punctuations = string.punctuation

# Define function to cleanup text by removing personal pronouns, stopwords, and puncuation
def cleanup_text(docs, logging=False):
    texts = []
    counter = 1
    for doc in docs:
        if counter % 1000 == 0 and logging:
            print("Processed %d out of %d documents." % (counter, len(docs)))
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in STOP_WORDS and tok not in punctuations]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)

# Clean up all text
data["claim"] = cleanup_text(data.claim, logging = False)

features = pre_process.transform(data.claim).toarray()

result = model.predict(features)
data['predict'] = list(result)

print('\nWriting predictions to:', PREDICTIONS_FILEPATH)
with open(PREDICTIONS_FILEPATH, 'w') as f:
    for ind in data.index:
        idx = data.loc[ind,'id']
        predict = data.loc[ind,'predict']
        f.write('%d,%d\n' % (idx, predict))
print('Finished writing predictions.')