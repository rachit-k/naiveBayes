import numpy as np
import csv
import matplotlib.pyplot as plt
import math

from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn import metrics

rows = []

vocab={}
n1=0
n2=0
p_stemmer = PorterStemmer()
tokenizer = TweetTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
punc_words=set([';',',','!','.',':'])
#stop_words.add('http')
#stop_words.add('com')

with open('train.csv', 'r',encoding='latin-1') as file: 

    data = csv.reader(file) 
  
    for row in data:         
#        print(row)
        pol=row[0]
        if pol=='2':
            continue
        no=row[1]
        dte=row[2]
        qry=row[3]
        usr=row[4]
        text=row[5]
        text = text.lower()
        text = tokenizer.tokenize(text)
        text = [ token for token in text if not token in stop_words ]
        text = [ p_stemmer.stem(token) for token in text ]
#        print(text)
        if pol=='0':
            n1=n1+1
            for word in text:
                if word in vocab:
                    vocab[word][0]=vocab[word][0]+1
                elif word!='': #and word[0]!='@':  
                    vocab[word]=[2,1]
#                else:
#                    print(word)
        else:
            n2=n2+1
            for word in text:
                if word in vocab:
                    vocab[word][1]=vocab[word][1]+1
                elif word!='':  
                    vocab[word]=[1,2]
#                else:
#                    print(word)               
            
#print(vocab)
n=n1+n2

ex=0
correct1=0
correct2=0
n1_test=0
n2_test=0
y_list=[]
scores=[]
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
        y_list.append(pol)    
        no=row[1]
        dte=row[2]
        qry=row[3]
        usr=row[4]
        text=row[5]
        text = text.lower()
        text = tokenizer.tokenize(text)
        text = [ token for token in text if not token in stop_words ]
        text = [ p_stemmer.stem(token) for token in text ]
#        print(text)        
        p1=math.log(n1/n)
        p2=math.log(n2/n)
        for word in text:
            if word in vocab:
                p1=p1+math.log(vocab[word][0])-math.log(vocab[word][0]+vocab[word][1])
                p2=p2+math.log(vocab[word][1])-math.log(vocab[word][0]+vocab[word][1])
#        print(p1)
#        print(p2) 
#        print(pol)
        scores.append(p2/(p1+p2))
        if p1>p2:
            if pol=='0':
                correct1=correct1+1
        else:
            if pol=='4':
                correct2=correct2+1
        ex=ex+1
accuracy=(correct1+correct2)*100/ex
print('accuracy:')  
print(accuracy)  
print('random accuracy:')  
print('50.0') 
print('majority accuracy:')  
print(max(n1_test,n2_test)*100/(n1_test+n2_test))

print('confusion matrix:')
print('actual')
print(correct1,n2_test-correct2)
print(n1_test-correct1,correct2)

fpr, tpr, thresholds = metrics.roc_curve(y_list, scores, pos_label='0')

print(fpr)
print(tpr)
print(thresholds)

plt.plot(fpr,tpr,label='roc')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.legend()
plt.savefig("Q1-g.png",bbox_inches="tight")
plt.show()

#accuracy:
#81.8941504178273
#random accuracy:
#50.0
#majority accuracy:
#50.69637883008357
#confusion matrix:
#actual
#144 32
#33 150
#[0.         0.00564972 0.13559322 0.13559322 0.24293785 0.24293785
# 0.42372881 0.42372881 0.47457627 0.47457627 0.48022599 0.48022599
# 0.50282486 0.50282486 0.51412429 0.51412429 0.53107345 0.53107345
# 0.5480226  0.5480226  0.55932203 0.55932203 0.61581921 0.61581921
# 0.64971751 0.64971751 0.66666667 0.66666667 0.6779661  0.6779661
# 0.69491525 0.69491525 0.71751412 0.71751412 0.73446328 0.73446328
# 0.75141243 0.75141243 0.75706215 0.75706215 0.76271186 0.76271186
# 0.79096045 0.79096045 0.79661017 0.79661017 0.80225989 0.80225989
# 0.81355932 0.81355932 0.81920904 0.81920904 0.82485876 0.82485876
# 0.83050847 0.83050847 0.83615819 0.83615819 0.85875706 0.85875706
# 0.8700565  0.8700565  0.87570621 0.87570621 0.88135593 0.88135593
# 0.88700565 0.88700565 0.89265537 0.89265537 0.9039548  0.9039548
# 0.91525424 0.91525424 0.92090395 0.92090395 0.92655367 0.92655367
# 0.93785311 0.93785311 0.94350282 0.94350282 0.94915254 0.94915254
# 0.95480226 0.95480226 0.96610169 0.96610169 0.97175141 0.97175141
# 0.97740113 0.97740113 0.98305085 0.98305085 0.98870056 0.98870056
# 0.99435028 0.99435028 1.        ]
#[0.         0.         0.         0.00549451 0.00549451 0.01098901
# 0.01098901 0.01648352 0.01648352 0.02197802 0.02197802 0.02747253
# 0.02747253 0.03296703 0.03296703 0.03846154 0.03846154 0.04395604
# 0.04395604 0.04945055 0.04945055 0.05494505 0.05494505 0.06593407
# 0.06593407 0.07142857 0.07142857 0.08241758 0.08241758 0.08791209
# 0.08791209 0.09340659 0.09340659 0.0989011  0.0989011  0.1043956
# 0.1043956  0.12637363 0.12637363 0.13186813 0.13186813 0.13736264
# 0.13736264 0.14285714 0.14285714 0.15934066 0.15934066 0.17032967
# 0.17032967 0.17582418 0.17582418 0.18131868 0.18131868 0.1978022
# 0.1978022  0.20879121 0.20879121 0.22527473 0.22527473 0.24175824
# 0.24175824 0.25824176 0.25824176 0.2967033  0.2967033  0.30769231
# 0.30769231 0.32417582 0.32417582 0.34065934 0.34065934 0.34615385
# 0.34615385 0.38461538 0.38461538 0.41758242 0.41758242 0.42307692
# 0.42307692 0.43956044 0.43956044 0.45054945 0.45054945 0.45604396
# 0.45604396 0.52747253 0.52747253 0.54945055 0.54945055 0.5989011
# 0.5989011  0.69230769 0.69230769 0.9010989  0.9010989  0.96153846
# 0.96153846 1.         1.        ]
#[1.76176568 0.76176568 0.65249641 0.65032049 0.61575289 0.61304222
# 0.58218784 0.5810613  0.57315195 0.57313595 0.57300595 0.5725906
# 0.5682547  0.56587717 0.56532546 0.56510577 0.56208624 0.56021289
# 0.55670103 0.55666739 0.55554133 0.55533168 0.54233927 0.54049893
# 0.5362054  0.5352754  0.53505178 0.53291339 0.53200671 0.53176437
# 0.53094779 0.52845528 0.52473222 0.52472279 0.5244372  0.52285993
# 0.51994681 0.51862917 0.51822084 0.51820912 0.51717624 0.51594659
# 0.5063495  0.50574117 0.50563512 0.50418177 0.50392525 0.50314673
# 0.50110946 0.50047631 0.49883855 0.49823434 0.49819192 0.49740039
# 0.49711792 0.49313644 0.49289473 0.49003861 0.4857555  0.48512086
# 0.48414337 0.48336227 0.48320597 0.47866934 0.47865925 0.47754188
# 0.47625775 0.47370968 0.47322667 0.47095377 0.47057248 0.46983745
# 0.46877717 0.46558367 0.46391868 0.45839499 0.45560078 0.45489014
# 0.45363807 0.45211538 0.45122347 0.44819367 0.44801793 0.44742741
# 0.44707502 0.43815919 0.43643391 0.43543464 0.4351042  0.42860751
# 0.4283319  0.41135772 0.41115355 0.36548065 0.36369201 0.33889009
# 0.33612243 0.31497474 0.16463354]