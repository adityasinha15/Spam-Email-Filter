# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 13:28:48 2019

@author: Aditya Sinha
"""
import os
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
import _pickle as c

direc = "emails/"
files = os.listdir(direc)
emails = [direc + email for email in files]
words = []
c = len(emails)
for email in emails:
    f = open(email, encoding='latin-1')
    blob = f.read()
    words += blob.split(" ")
    print(c)
    c -= 1

for i in range(len(words)):
    if not words[i].isalpha():
        words[i] = ""

dictionary = Counter(words)
del dictionary[""]