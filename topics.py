
import sys
sys.path.insert(0, "./package")


import pickle
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import gensim.corpora
from gensim.models import ldamodel
from nltk.corpus import stopwords
import nltk
import sklearn
import scipy as sp
import numpy as np
import pandas as pd

nltk.download('stopwords')


def printDataFrame(data):
    if(type(data) is pd.DataFrame):
        for idx in range(100):
            print(data.iloc[idx]['headline_text'])
    if(isinstance([10, 20, 30], data)):
        for idx in range(100):
            print(data[idx])

def get_lda_topics(model, num_topics):
    word_dict = {}
    for i in range(num_topics):
        words = model.show_topic(i, topn = 20)
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words]
    return pd.DataFrame(word_dict)

try:
        data_text = pickle.load(open('data_text.dat', 'rb'))
except:
        data_text = None

if data_text is None:

    data = pd.read_csv('data/abcnews-date-text.csv', error_bad_lines=False)

    # We only need the Headlines text column from the data
    data_text = data[['headline_text']]

    data_text = data_text.astype('str')

    for idx in range(len(data_text)):
        # go through each word in each data_text row, remove stopwords, and set them on the index.
        data_text.iloc[idx]['headline_text'] = [word for word in data_text.iloc[idx]
                                                ['headline_text'].split(' ') if word not in stopwords.words()]

    # print logs to monitor output
    if idx % 1000 == 0:
        sys.stdout.write('\rc = ' + str(idx) + ' / ' + str(len(data_text)))
    # save data because it takes very long to remove stop words
    pickle.dump(data_text, open('data_text.dat', 'wb'))
# get the words as an array for lda input
# train_headlines = [value[0] for value in data_text.iloc[0:].values] # train_headlines = [['aba','decides','community','breadcasting','license'],['act','fire'...]...]
 
# num_topics= 10 # in LDA we provide it a total number of topics that we want to derieve. 

# id2word = gensim.corpora.Dictionary(train_headlines);
# corpus = [id2word.doc2bow(text) for text in train_headlines];
# lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics);

train_headlines = [['I','fool'],['nasa', 'scientists', 'fool', 'appolo', 'spaceship', 'mars', 'fool']]

num_topics= 2

id2word = gensim.corpora.Dictionary(train_headlines)
# This takes the whole set of words and assigns an int id (token) to each word starting with 0
# example: 
#   {'I': 0, 'appolo': 2, 'fool': 1, 'mars': 3, 'nasa': 4, 'sceintists': 5, 'spaceship': 6}
corpus = [id2word.doc2bow(text) for text in train_headlines]
# now again we iterate through each document (each inner array) and then assigns count to the token for each word
# corpus = [ 
#       [(0,1),(1,1)],
#       [(1, 2), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]
# ]
lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)

word_dict = get_lda_topics(lda, num_topics)

print(word_dict)