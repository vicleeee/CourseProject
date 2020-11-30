# CourseProject

Please fork this repository and paste the github link of your fork on Microsoft CMT. Detailed instructions are on Coursera under Week 1: Course Project Overview/Week 9 Activities.

Stage 1 By Nov 29. 
Reproduce of Step 5.1

file: test.py  

First we remove the reviews with any missing aspect rating or document length less than 50 words (to keep the content coverage of all possible aspects).  
Then we  convert all the words into lower cases and remove punctuations and stop words.  
In vocab.txt we write vocabulary appearance based on reviews. If a word appears in several times in the same review, it would only be counted as once.  We then filtered out words that have less than ten occurences.  

Step 5.2 In Progress  
