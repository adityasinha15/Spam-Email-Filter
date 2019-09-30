# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 17:43:30 2019

@author: Aditya Sinha
"""

import os
from collections import Counter

def make_dict():
    direc = 'emails/'
    files = os.listdir(direc)
    emails = [direc + email for email in files]
    words = []
    for email in emails:
        f= open(email, encoding='latin-1')
        blob = f.read()
        words += blob.split(" ")
        
    for i in range(len(words)):
        if not words[i].isalpha():
            words[i] = ""
    
    dictionary = Counter(words)
    del dictionary[""]
    return dictionary
    
def make_dataset(dictionary):
    direc = 'emails/'
    files = os.listdir(direc)
    emails = [direc + email for email in files]
    features = []
    labels = []
    for email in emails:
         f= open(email, encoding='latin-1')
         words += f.read().split(" ")
         data = []
         for entry in dictionary:
             data.append(words.count(entry[0]))
        
         features.append(data)
         
         if "ham" in emails:
             labels.append(0)
         if "spam" in emails:
             labels.append(1)
d = make_dict()
make_dataset(d)   
        
         
    
    
