import chardet
file = "/content/spam.csv"
with open(file,'rb') as rawdata:
	result = chardet.detect(rawdata.read(10000))
result
import pandas as pd
dataset = pd.read_csv("/content/spam.csv",encoding="windows-1252")
dataset.head()
dataset.info()
dataset.isnull().sum()
x=dataset["v1"].values
y=dataset["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer() 
x_train=cv.fit_transform(x_train) 
x_test=cv.transform(x_test) 
from sklearn.svm import SVC 
svc=SVC() 
svc.fit(x_train,y_train) 
y_pred=svc.predict(x_test) 
y_pred
from sklearn import metrics 
accuracy=metrics.accuracy_score(y_test,y_pred) 
accuracy