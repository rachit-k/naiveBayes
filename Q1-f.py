import numpy as np
import csv
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
import timeit
from sklearn.feature_selection import SelectPercentile, chi2

rows = []

vocab={}

n1=0
n2=0
X_train=[]
y_train=[]
k=-1
with open('train.csv', 'r',encoding='latin-1') as file: 
    data = csv.reader(file)  
    for row in data: 
#        print(row)
        pol=row[0]
        if pol=='2':
            continue
#        k=k+1
#        if k%10!=0:
#            continue
        no=row[1]
        dte=row[2]
        qry=row[3]
        usr=row[4]
        text=row[5]
        X_train.append(text)
        y_train.append(pol)
        if pol=='0':
            n1=n1+1
        else:
            n2=n2+1

# X_train = SelectPercentile(chi2, percentile=10).fit_transform(X_train, y_train)         
# print(X_train)
#X_train=(np.split(X_train,1000))[0]
#y_train=(np.split(y_train,1000))[0]
n=n1+n2

n1_test=0
n2_test=0

X_test=[]
y_test=[]

with open('test.csv', 'r',encoding='latin-1') as file1: 
    data = csv.reader(file1)   
    for row in data:         
        pol=row[0]
        if pol=='2':
            continue
        elif pol=='0':
            n1_test=n1_test+1
        else:
            n2_test=n2_test+1
        no=row[1]
        dte=row[2]
        qry=row[3]
        usr=row[4]
        text=row[5]
        X_test.append(text)
        y_test.append(pol)

starttime = timeit.default_timer()

vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
X_train_vec = SelectPercentile(chi2, percentile=0.5).fit_transform(X_train_vec, y_train)
X_test_vec = SelectPercentile(chi2, percentile=0.5).fit_transform(X_test_vec, y_test)

nb = GaussianNB()
y_train=np.array(y_train)


X_train_vec1=(X_train_vec.todense())
X_test_vec1=X_test_vec.todense()

#X_train_vec1=X_train_vec1[:,:-1]
nb.fit(X_train_vec1, y_train)
#print(X_train_vec1.shape)

#l=np.unique(y_train)
#for i in range(10000):
#    y_train1=y_train[i*160:(i+1)*160]
#    X_train_vec1=(X_train_vec[i*160:(i+1)*160,:]).todense()   
#    nb.partial_fit(X_train_vec1, y_train1,l)

y_pred_class = nb.predict(X_test_vec1)

correct=0.0
for i in range(n1_test+n2_test):
    if y_test[i]==y_pred_class[i]:
        correct=correct+1
        
print("time:")    
print(timeit.default_timer()-starttime)
print("accuracy:")
print(correct*100/(n1_test+n2_test))

print('random accuracy:')
print('50.0')
print('majority accuracy:')  
print(max(n1_test,n2_test)*100/(n1_test+n2_test))

