 # -*- coding: utf-8 -*-
'''
CS-IS-1
Osama Shahat       20160046    
Abdelrahman Kamal  20160133
Shady Aasim        20160112    
'''
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split 
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout,Flatten 
from sklearn.metrics import accuracy_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from keras.layers import Lambda
from keras import backend as K
import pickle

#read the file and return dataset ,text and airline_sentiment
def readFile(x):
    columns = ["airline_sentiment", "text"]
    dataSet=pd.read_csv(x,usecols=columns)
    y = dataSet['airline_sentiment']    
    x =dataSet['text']                  
    return dataSet,x,y

#take each text in text column and remove -> 
def cleanData(tweet_Review):
    tweet_Review = re.sub(r'@\w+', '',tweet_Review)            # remove mention from text  
    tweet_Review = re.sub(r'[^a-zA-Z0-9\s]','',tweet_Review)   #remove non alphabetical and numerical
    tweet_Review = re.sub(r'http\w+', '',tweet_Review)         # remove url from text 
    tweet_Review = tweet_Review .lower()                      # make all text in lowercase
    tweet_Review=[word          
    for word in tweet_Review.split()
        if not word in set(stopwords.words('english'))]      #remove stopwords from text
    tweet_Review=" ".join(tweet_Review)
    return tweet_Review

                        #the max of index    [0]    [1]    [2]
#obtain the which sentiment is majority (negative,neutral,positive) and convert 
#this method to facilitate obtain the Accuracy
def Labeling(Y):
    Y_label=[]
    for i in range (0,len(Y)):
        if np.argmax(Y[i])==0:
            Y_label.append(0)
        elif np.argmax(Y[i])==1:
            Y_label.append(1)
        elif np.argmax(Y[i])==2:
            Y_label.append(2)
    Y_label=np.array(Y_label,dtype=np.int64)
    return Y_label
#convert label from numerical to alphabetical
def getLabel(Y):
    label=Labeling(Y)
    if label==0:
        return "Negative"
    elif label==1:
        return "Neautral"
    else:
        return "Positive"

corpus=[]
dataSet,X,Y=readFile('Tweets.csv')
for i in range(0,len(dataSet['text'])):
    corpus.append(cleanData(dataSet["text"][i]))

tokenizer = Tokenizer(5000)
tokenizer.fit_on_texts(corpus)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(corpus)
X = pad_sequences(sequences, maxlen=33)   

Y = pd.get_dummies(dataSet['airline_sentiment']).values

def mean(x, axis):
  """mean
     input_shape=(batch_size, time_slots, ndims)
     depending on the axis mean will:
       0: compute mean value for a batch and reduce batch size to 1
       1: compute mean value across time slots and reduce time_slots to 1
       2: compute mean value across ndims an reduce dims to 1.
  """
  return K.mean(x, axis=axis, keepdims=True)

model = Sequential()
model.add(Embedding(5000,256, input_length=X.shape[1]))
model.add(Dropout(0.3))
model.add(Lambda(lambda x: mean(x,axis=1)))
#model.add(Lambda(lambda x : K.sum(x) ))#, output_shape=None))
#model.add(Flatten())
model.add(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
model.add(LSTM(256, dropout=0.3, recurrent_dropout=0.2))
model.add(Dense(3, activation='sigmoid'))    
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

model.fit(X_train, Y_train, epochs=8, batch_size=32, verbose=2)

#to save model
pickle.dump(model, open('kerasMeanModel.sav', 'wb'))

Y_predict=model.predict(X_test)
Y_testing=Labeling(Y_test)
Y_predicting=Labeling(Y_predict)
print("The Accuracy for LSTM( Avarage) model is ",accuracy_score(Y_testing, Y_predicting)*100)

while True:
    x=str(input("Enter your sentence to check her sentiment,if you want to exit 'Exit'\n"))
    if x == 'Exit' or x=='exit':
        break
    x=cleanData(x)
    sequence = tokenizer.texts_to_sequences([x])
    x = pad_sequences(sequence, maxlen=33)
    y=model.predict(x)
    print(getLabel(y))