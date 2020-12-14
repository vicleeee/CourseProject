# CourseProject

Course Project for CS 410: Reproducing a paper, Latent Aspect Rating Analysis without Aspect Keyword Supervision. 
Paper link: https://www.cs.virginia.edu/~hw5x/paper/p618.pdf


**Stage 1 By Nov 29**  
Reproduce of Step 5.1

*file: test.py*  

First we remove the reviews with any missing aspect rating or document length less than 50 words (to keep the content coverage of all possible aspects).  
Then we  convert all the words into lower cases and remove punctuations and stop words.  
In vocab.txt we write vocabulary appearance based on reviews. If a word appears several times in the same review, it would only be counted as once.  We then filtered out words that have less than ten occurrences.  

**Step 5.2 In Progress**  


**Documentation**
**1.Overview**

This project consists of tasks of preprocessing data and implementing LARA functions. 
We get the data from  http://timan.cs.uiuc.edu/ downloads.html and we focused on TripAdvisor data for this project. 

**2.Programming Language and Packages**

Python 3.X
Packages: numpy, scipy, math, re, random, nltk

**3.Implementation**

Clean.py
This is the python program for preprocessing the data, we did the following for this part:  1) remove the reviews with any missing aspect rating or document length less than 50 words (to keep the content coverage of all possible aspects); 2) convert all the words into lower cases; and 3) removing punctuations, stop words, and the terms occurring in less than 10 reviews in the collection. 

Lara.py
	This is the main program we implemented all the functions for building this LARA model. 
	In this program, we implemented function such as update_mu, update_beta, E_step, 
	M_step etc. 
	
Load.py
This is the python program we have to load our data and build our vocabulary. 


**4.Project Members**

Ziyuan Wei (ziyuan3@illinois.edu)
Xinyi He (xinyihe4@illinois.edu)
Weijiang Li (wl13@illinois.edu)
Dingsen Shi (dingsen2@illinois.edu)
Qunyu Shen (qunyus2@illinois.edu)

We decided to collaborate with another team, led by Xinyi  He, half way through the project since we met some challenges when understanding the methods used in the paper. Then we splitted our tasks between two groups, our group (led by Weijiang Li, collaborating with Ziyuan Wei) focuses on the implementation of preprocessing and EM steps in the building up the model, and the other group contributed to the rest of the functions such as negative likelihood, and another main part of this project is the implementation of bootstrap. Team members from both team worked hard to try to get the code done based on the method description from the paper. 

When implementing EM steps, we separate the procedure into two functions, e-step() and m-step(), before adopting a new function runEM() to combine and output the previous data. 

**5. Video Link**

Here is the link to the demo video on mediaspace:
https://mediaspace.illinois.edu/media/t/1_fo2gtfej
