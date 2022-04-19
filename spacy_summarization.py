# NLP Pkgs
import spacy
nlp = spacy.load("en_core_web_sm")

#nlp = spacy.load('en')
# Pkgs for Normalizing Text
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
# Import Heapq for Finding the Top N Sentences
from heapq import nlargest

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

stopwords=list(STOP_WORDS)



def text_summarizer(raw_docx):
    doc=nlp(raw_docx)
    tokens=[token.text for token in doc]
    #print(tokens)
    word_freqtable = {}
    for word in doc:
        if stemmer.stem(word.text.lower()) not in stopwords:
            if stemmer.stem(word.text.lower()) not in punctuation:
                if stemmer.stem(word.text.lower()) not in word_freqtable.keys():
                    word_freqtable[stemmer.stem(word.text.lower())]=1
                else:
                    word_freqtable[stemmer.stem(word.text.lower())]+=1
    sentence_tokens=[sent for sent in doc.sents]
    #sentence_tokens
    sentence_scores={}
    for sent in sentence_tokens:
        for word in sent:
            if stemmer.stem(word.text.lower()) in word_freqtable.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent]=word_freqtable[stemmer.stem(word.text.lower())]
                else:
                    sentence_scores[sent]+=word_freqtable[stemmer.stem(word.text.lower())]
    sum_size= int(len(sentence_tokens)*0.25)
    summary=nlargest(sum_size,sentence_scores,key=sentence_scores.get)
    final_summary=[word.text for word in summary]
    summary=' '.join(final_summary)
    return summary
