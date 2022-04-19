import pandas as pd
import numpy as np
import nltk
import spacy
nlp=spacy.load('en_core_web_sm')

import re
from nltk.tokenize import sent_tokenize
#nltk.download('stopwords')  # one time execution
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import distance


def kk_summarizer(raw_text):
    sentence = sent_tokenize(raw_text)
    corpus = []
    for i in range(len(sentence)):
        sen = re.sub('[^a-zA-Z]', " ", sentence[i])
        sen = sen.lower()
        sen = sen.split()
        sen = ' '.join([i for i in sen if i not in stopwords.words('english')])
        if len(sen)>0:
          corpus.append(sen)
    all_words = [i.split() for i in corpus]
    model = Word2Vec(all_words, min_count=1)

    sent_vector=[]
    for i in corpus:
        plus=0
        for j in i.split():
            plus+= model.wv[j]
        plus = plus/len(i.split())

        sent_vector.append(plus)

    pca = PCA(2)
    df = pca.fit_transform(sent_vector)
    thres_size= int(len(sentence)*0.25)

    n_clusters = thres_size
    kmeans = KMeans(n_clusters, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(df)


    centroids = kmeans.cluster_centers_
    u_labels = np.unique(y_kmeans)

    my_list=[]
    for i in range(n_clusters):
        my_dict={}

        for j in range(len(y_kmeans)):

            if y_kmeans[j]==i:
                my_dict[j] =  distance.euclidean(kmeans.cluster_centers_[i],df[j])
        min_distance = min(my_dict.values())
        my_list.append(min(my_dict, key=my_dict.get))

        summary=[]
    for i in sorted(my_list):
        summary.append(sentence[i])

    summ= ' '.join([i for i in summary])
    return summ
