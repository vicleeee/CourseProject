from collections import defaultdict
from datetime import datetime
import random
import nltk
from nltk.corpus import stopwords
import json
import os
import shutil
import string


# data = '/'.join([os.getcwd(), 'raw_data'])

stop_words = set(stopwords.words('english'))
ids = set([])

cnt = 0
curr_data = os.listdir('data')
test_data = curr_data[:100]
# print(test_data)

reviews = []
for name in test_data:
  tmp = []
  with open('/'.join(['data', name])) as fp:
    for review in json.load(fp)['Reviews']:
      tmp.append(review)
  reviews.append(tmp)

word_dict = {}
new_reviews = []
for tmp in reviews:
  for review in tmp:
    if len(review['Ratings']) < 7:
      tmp.remove(review)
    elif len(review['Content']) < 50:
      tmp.remove(review)
    # get rid of punctuations
    review['Content'] = review['Content'].translate(str.maketrans('', '', string.punctuation))
    # lowercase
    review['Content'] = review['Content'].lower()
    tmp2 = []
    for word in review['Content'].split(' '):
      if word  == '' or word in stop_words:
        continue
      tmp2.append(word)
      word_dict[word] = word_dict.get(word, 0) + 1
    review['Content'] = ' '.join(tmp2)
    new_reviews.append(review['Content'])
    # print(review['Content'])

cnt = 0
for tmp in reviews:
  for review in tmp:
    tmp2 = []
    for word in new_reviews[cnt].split(' '):
      # print(word)
      if word not in word_dict or word_dict[word] < 10:
        continue
      tmp2.append(word)
    review['Content'] = ' '.join(tmp2)
    # print(review['Content'])
    cnt += 1
# print(word_dict)


fp = open('tmp.txt', 'r')
lines = fp.readlines()
arr = []
for line in lines:
  for word in line.split(' '):
    arr.append(word)
# print(arr)

# fp = open('reviews.txt', 'w')
# for tmp in reviews:
#   for review in tmp:
#     fp.write(review['Content'] + '\n')


cnt = 1
shutil.rmtree('result')
os.mkdir('result')
os.chdir('result/')
for tmp in reviews:
  fp = open(str(cnt) + '.json', 'w')
  fp.write('{')
  fp.write('\"Reviews\"' + ': [')
  for review in tmp:
    json.dump(review, fp, default=str)
    fp.write(',\n')
  fp.write(']}')
  cnt += 1
