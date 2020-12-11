from collections import defaultdict
from datetime import datetime
import json
import os
import shutil
import string


curr_dir = os.path.dirname(os.path.realpath(__file__))
data = '/'.join([curr_dir, 'Data', 'raw_data'])

stop_words = ''
with open('/'.join([curr_dir, 'Data', 'stop_words.json'])) as stop_words:
  stop_words = set(json.load(stop_words))

ids = set([])

cnt = 0
curr_data = os.listdir(data)
test_data = curr_data[:10]
# print(test_data)

reviews = []
for name in test_data:
  with open('/'.join([data, name])) as fp:
    for review in json.load(fp)['Reviews']:
      reviews.append(review)

'''
Remove reviews with any missing aspect rating or document
length less than 50 words

Convert all the words into lower cases

Remove punctuation and stop words
Gather -- Remove words occuring in less than 10 reviews in the collection
'''

word_dict = {}
new_reviews = []
for review in reviews:
  if len(review['Ratings']) < 7:
    reviews.remove(review)
  elif len(review['Content']) < 50:
    reviews.remove(review)
  review['Content'] = review['Content'].translate(str.maketrans('', '', string.punctuation))
  review['Content'] = review['Content'].lower()
  tmp = []
  for word in review['Content'].split(' '):
    if word  == '' or word in stop_words:
      continue
    tmp.append(word)
    word_dict[word] = word_dict.get(word, 0) + 1
  review['Content'] = ' '.join(tmp)
  new_reviews.append(review['Content'])
  # print(review['Content'])
print(len(reviews))

cnt = 0
for review in reviews:
  tmp = []
  for word in new_reviews[cnt].split(' '):
    # print(word)
    if word not in word_dict or word_dict[word] < 10:
      continue
    tmp.append(word)
  review['Content'] = ' '.join(tmp)
  print(review['Content'])
  cnt += 1

fp = open('result.json', 'w')
fp.write('[')
for review in reviews:
  # print(review)
  json.dump(review, fp, default=str)
  fp.write(',\n')
fp.write(']')
