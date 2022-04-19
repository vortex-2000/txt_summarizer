from __future__ import unicode_literals
from flask import Flask,render_template,url_for,request


#from spacy_summarization import text_summarizer
from gensim.summarization import summarize
from nltk_summarization import nltk_summarizer
from kk_means import kk_summarizer
import time
import spacy
nlp = spacy.load("en_core_web_sm")
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
from heapq import nlargest
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

app = Flask(__name__)

# Web Scraping Pkg
from bs4 import BeautifulSoup
# from urllib.request import urlopen
from urllib.request import urlopen

# Sumy Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer


#Statistical
def text_summarizer(docx):
	stopwords=list(STOP_WORDS)
	doc=nlp(docx)
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


# Sumy
def sumy_summary(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	doc=nlp(docx)
	sentence_tokens=[sent for sent in doc.sents]
	thres_size= int(len(sentence_tokens)*0.25)
	summary = lex_summarizer(parser.document,thres_size)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result


def lsa_summary(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lsa_summarizer=LsaSummarizer()
	doc=nlp(docx)
	sentence_tokens=[sent for sent in doc.sents]
	thres_size= int(len(sentence_tokens)*0.25)
	summary = lsa_summarizer(parser.document,thres_size)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result


def tex_summary(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	doc=nlp(docx)
	sentence_tokens=[sent for sent in doc.sents]
	thres_size= int(len(sentence_tokens)*0.25)
	summarizer = TextRankSummarizer()
	Tex_summary = summarizer(parser.document,thres_size)
	result=' '.join([str(item) for item in Tex_summary] )
	return result

# Reading Time
def readingTime(mytext):
	total_words = len([ token.text for token in nlp(mytext)])
	estimatedTime = total_words/200.0
	return estimatedTime

# Fetch Text From Url
def get_text(url):
	page = urlopen(url)
	soup = BeautifulSoup(page)
	fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
	return fetched_text

@app.route('/')
def index():
	return render_template('index.html')


@app.route('/analyze',methods=['GET','POST'])
def analyze():
	start = time.time()
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		final_reading_time = readingTime(rawtext)
		final_summary = text_summarizer(rawtext)
		summary_reading_time = readingTime(final_summary)
		end = time.time()
		final_time = end-start
	return render_template('index.html',ctext=rawtext,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)

@app.route('/analyze_url',methods=['GET','POST'])
def analyze_url():
	start = time.time()
	if request.method == 'POST':
		raw_url = request.form['raw_url']
		rawtext = get_text(raw_url)
		final_reading_time = readingTime(rawtext)
		final_summary = text_summarizer(rawtext)
		summary_reading_time = readingTime(final_summary)
		end = time.time()
		final_time = end-start
	return render_template('index.html',ctext=rawtext,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)



@app.route('/compare_summary')
def compare_summary():
	return render_template('compare_summary.html')

@app.route('/comparer',methods=['GET','POST'])
def comparer():
	start = time.time()
	if request.method == 'POST':
		#Statistical spacy
		rawtext = request.form['rawtext']
		final_reading_time = readingTime(rawtext)
		final_summary_spacy = text_summarizer(rawtext)
		summary_reading_time = readingTime(final_summary_spacy)
		# Gensim
		# texttrank
		# final_summary_gensim = summarize(rawtext)							##
		# summary_reading_time_gensim = readingTime(final_summary_gensim)
		final_summary_gensim = tex_summary(rawtext)							##
		summary_reading_time_gensim = readingTime(final_summary_gensim)

		# NLTK # Lsa sumy
		#final_summary_nltk = nltk_summarizer(rawtext)
		#summary_reading_time_nltk = readingTime(final_summary_nltk)
		final_summary_lsa = lsa_summary(rawtext)
		summary_reading_time_lsa = readingTime(final_summary_lsa)

		# Lexrank Sumy
		final_summary_sumy = sumy_summary(rawtext)							##
		summary_reading_time_sumy = readingTime(final_summary_sumy)


		#nltk tfidf
		final_summary_nltk = nltk_summarizer(rawtext)							##
		summary_reading_time_nltk = readingTime(final_summary_nltk)


		# KKMEANS
		final_summary_kk = kk_summarizer(rawtext)							##
		summary_reading_time_kk = readingTime(final_summary_kk)

		end = time.time()
		final_time = end-start
	return render_template('compare_summary.html',ctext=rawtext,final_summary_spacy=final_summary_spacy,final_summary_gensim=final_summary_gensim,final_summary_lsa=final_summary_lsa,final_summary_nltk=final_summary_nltk,final_summary_kk=final_summary_kk,    final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time,summary_reading_time_gensim=summary_reading_time_gensim,final_summary_sumy=final_summary_sumy,summary_reading_time_sumy=summary_reading_time_sumy,summary_reading_time_lsa=summary_reading_time_lsa,summary_reading_time_nltk=summary_reading_time_nltk,summary_reading_time_kk=summary_reading_time_kk)



@app.route('/about')
def about():
	return render_template('index.html')

if __name__ == '__main__':
	app.run(debug=True)