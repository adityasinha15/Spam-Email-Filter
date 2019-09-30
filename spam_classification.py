# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 22:08:44 2019

@author: Aditya Sinha
"""
import numpy as np
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt

direc = 'emails/'
files = os.listdir(direc)
emails_list = [direc + email for email in files]
emails = []
labels = []
for email in emails_list:
    f = open(email, encoding='latin-1')
    emails.append(f.read())
    if "ham" in email:
        labels.append(0)
    if "spam" in email:
        labels.append(1)
    
dataset = pd.DataFrame()
dataset['email'] = emails
dataset['label'] = labels
corpus = []
ps = PorterStemmer()
for i in range(0, 5172):
    email = re.sub('[^a-zA_Z]', ' ', dataset['email'][i])
    email = email.lower()
    email = email.split()
    email = [ps.stem(word) for word in email if not word in stopwords.words('english')]
    email = ' '.join(email)
    corpus.append(email)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
cv1 = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
X1 = cv.fit_transform(['hello you won RS2000', '200 won by you']).toarray()
Y = dataset.iloc[:, 1]

from sklearn.cross_validation import train_test_split
X_train, X_test,  Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, Y_train)
Y_pred = nb.predict(X_test)
pred = nb.predict(X1)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

objects = ['Spam', 'Ham']
y_pos = np.arange(len(objects))
performance = []
y = np.array(Y_pred)
spam = np.count_nonzero(y == 1)
ham = np.count_nonzero(y == 0)
performance.append(spam)
performance.append(ham)
plt.bar(y_pos, performance, align='center', alpha=1, color=['r', 'g'])
plt.xticks(y_pos, objects)
plt.ylabel('Number of Responses')
plt.title('Spam VS Ham')
plt.show()




    
    


    