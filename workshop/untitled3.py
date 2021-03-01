import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

dataset=pd.read_csv('SMSSpamCollection',sep='\t',names=['label','msg'])

stemmer=PorterStemmer()
corpus=[]
for i in range(len(dataset)):
    words=re.sub("^[a-zA-Z]",' ',dataset['msg'][i])
    words=words.lower()
    words=words.split()
    words=[stemmer.stem(word) for word in words if not word in stopwords.words('english')]
    words=" ".join(words)
    corpus.append(words)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2500)
x=cv.fit_transform(corpus).toarray()


y=pd.get_dummies(dataset['label'])
y=y.iloc[:,1].values


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)

from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()

model.fit(x_train,y_train)

y_predicted=model.predict(x_test)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_predicted)
import seaborn as sn
sn.head(cm,annot=True)




