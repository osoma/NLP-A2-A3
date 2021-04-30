# -*- coding: utf-8 -*-
'''
CS-IS-1
Osama Shahat       20160046    
Abdelrahman Kamal  20160133
Shady Aasim        20160112    
'''
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#read the file
dataset=pd.read_csv('Tweets.csv')
y = dataset.iloc[: , 1]    # airline_sentiment

#cleaning the text, to remove the stopwords and the spaicial chars
corpus=[]
for i in range(0,len(dataset['text'])):
    tweet_Review = re.sub('[^a-zA-Z]', ' ',dataset['text'][i])
    tweet_Review = tweet_Review .lower()
    tweet_Review =tweet_Review .split()
    #NLTK stemming->for get the root of the word  for example (loved will be love)
    ps= PorterStemmer()
    tweet_Review=[ps.stem(word)
    for word in tweet_Review 
        if not word in set(stopwords.words('english'))]
    tweet_Review=' '.join(tweet_Review) #to make it a simple sentace again
    corpus.append(tweet_Review)

#make each label from airline_sentiment to fixed number
le_y= LabelEncoder()
y=le_y.fit_transform(y)  #0--> negative , 1--> neutral , 2-->Positive

count_vect = CountVectorizer(max_features=10000)
corpus=count_vect.fit_transform(corpus).toarray()
#data spiltting
X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size = 0.2, random_state = 4)

#count_vect = CountVectorizer()
#X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train)
clf = MultinomialNB().fit(X_train_tfidf, y_train)
y_predict=clf.predict(X_test)

print("The Accuracy for this model is ",(accuracy_score(y_test, y_predict))*100)
  #-#____________
from sklearn.externals import joblib
joblib.dump(clf, 'model.pkl') # to save the classifier 
#--------------------------
classifr=joblib.load('model.pkl')
input_text='osama it is good and nice'
test_df = pd.Series(data=input_text)

cv = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
cv.fit(dataset['text'])
texttt=cv.transform(test_df)
Z=classifr.predict(texttt)
print(Z)
#------
text_review=input("Enter a new text review ")
text_review=clf.predict(count_vect.fit_transform(text_review))
print(clf.predict(text_review))

"""
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

print("Accuracy:", score)
"""