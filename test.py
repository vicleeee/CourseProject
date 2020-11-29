import os, sys, string
import nltk
from nltk import *

# remove the reviews with any missing aspect rating
# or document length less than 50 words (to keep the content
# coverage of all possible aspects)
def load_file(file):
	reviews, ratings = [], []
	f = open(file,'r')
	flag = False

	for line in f:
		curr = line.strip().split('>')
		
		first_w = curr[0]
		curr_review, curr_rating = "", []
		if first_w == '<Content':
			flag = False
			curr_review = str(curr[1])
			if len(curr_review) < 50:
				continue
			flag = True
		elif first_w == '<Rating':
			# seven aspects
			if flag == False:
				continue
			flag = True
			curr_rating = curr[1].split('\t')
			if -1 in curr_rating:
				continue
		if flag:
			if curr_review:
				reviews.append(curr_review)
			if curr_rating:
				ratings.append(curr_rating)
	f.close()
	return reviews, ratings

def read_stop_words(file):
	stop_words = []
	f = open(file,'r')
	cnt = 0
	for line in f:
		if cnt % 2 == 0:
			stop_words.append(line.strip('\n'))
		cnt += 1
	return stop_words

# convert all the words into lower cases
# removing punctuations, stop words
def parse_to_sentence(reviews):
	res = []
	# remove stopwords
	for review in reviews:
		sentences = nltk.sent_tokenize(review)
		curr = []
		for sentence in sentences:
			tmp = sentence.lower()
			# remove punctuations
			for ch in tmp:
				if ch in punctuations:
					tmp = tmp.replace(ch, "")
			# remove stopwords
			arr = tmp.split(" ")
			for word in arr:
				if word in stop_words or not word:
					arr.remove(word)
			curr.append(tmp)
		res.append(curr)
	return res
	
stop_words = read_stop_words('./stopwords.txt')
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
dir = './Aspects/'

fp = open('words.txt','a+')
for f_name in os.listdir(dir):
	reviews, ratings = load_file(dir + f_name)
	processed_res = parse_to_sentence(reviews)
	total, num = 0, 0
	for sentences in processed_res:
		total += 1
		for sentence in sentences:
			num += 1
			fp.write(sentence)
		fp.write('\n')

# remove terms occurinng inn less than 10 reviews
def check_freq(arr):
	word_dict = {}
	for sentence in arr:
		for word in sentence:
			word_dict[word] = word_dict.get(word) + 1
			
# build_vocab(processed_res)

