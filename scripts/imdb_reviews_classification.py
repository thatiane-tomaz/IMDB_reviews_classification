#importing libraries

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import joblib
from datetime import datetime

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, roc_auc_score

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
%matplotlib inline




# Preprocessing

df = pd.read_csv('IMDB Dataset.csv')

#normalizing to lowercase letters
df['review'] = df['review'].str.lower()

#removing hyperlinks
df['review'] = df['review'].str.replace('[^\w\s]','')

#tokenization 
df['review_tok'] = df['review'].str.split()

#removing stopwords
stop_words = set(stopwords.words('english'))
df['review_stop'] = df['review_tok'].apply(lambda x: [item for item in x if item not in stop_words])
  
#Stemming - Porter Stemmer
porter_stemmer = PorterStemmer()
df['review_PS'] = df['review_stop'].apply(lambda x: [porter_stemmer.stem(y) for y in x])



# Bag of Words - TF-IDF

X = df['review_PS'].map(' '.join)
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features = 500000)

# Fit the model
tf_transformer = tf.fit(X)

# Dump the file
pickle.dump(tf_transformer, open("tfidf1_v3.pkl", "wb"))

X_tf = tf_transformer.fit_transform(X)


# Training Model - Random Forest

y = df['sentiment']

# split train/test

X_train,X_test,y_train,y_test = train_test_split(X_tf, y, test_size=0.2, random_state=0)

print('training model...')
# train test classification

classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train, y_train)



# Testing model and analyzing the results

# predict
y_pred = classifier.predict(X_test)
y_score = classifier.predict_proba(X_test)[:,1]
print(classification_report(y_test,y_pred))
print("Accuracy score:")
print(accuracy_score(y_test, y_pred))
print("AUC score:")
print(roc_auc_score(y_test, y_score))


