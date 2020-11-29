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

def count_word_in_reivew(review):
	word_list = set([])
	for sentence in review:
		words = sentence.split(" ")
		for word in words:
			if word not in word_list:
				word_list.add(word)
	# for every review
	for word in word_list:
		if word not in total_freq:
			total_freq[word] = 1
		else:
			total_freq[word] += 1

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

		count_word_in_reivew(curr)

	return res

# remove terms occurinng in less than 10 reviews
def filter_words(reviews):
	res = []
	for review in reviews:
		curr = []
		for sentence in reviews:
			words = sentence.split(" ")
			for word in words:
				if word not in total_freq or total_freq[word] < 10:
					words.remove(word)
			curr.append(''.join(words))
		res.append(curr)
	return res

stop_words = read_stop_words('./stopwords.txt')
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
dir = './Aspects/'
total_freq = {}		# words with occurrences in reviews

# fp = open('words.txt','a+')
fp2 = open('vocab.txt', 'w')
cnt = 0
for f_name in os.listdir(dir):
	reviews, ratings = load_file(dir + f_name)
	processed_res = parse_to_sentence(reviews)
# print(total_freq)

# write total word occurrence
for word in total_freq:
	fp2.write(str(word) + ' ' + str(total_freq[word]) + '\n')

fp3 = open('final_reivews.txt', 'a+')
for f_name in os.listdir(dir):
	reviews, ratings = load_file(dir + f_name)
	final_reviews = filter_words(reviews)
	for review in final_reviews:
		for sentence in review:
			fp3.write(sentence + '\n')
	fp3.write('\n' + '\n' + '\n')
		

