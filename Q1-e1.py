import csv
import math
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re



rows = []

vocab={}
# reading csv file 
n1=0
n2=0
p_stemmer = PorterStemmer()
tokenizer = TweetTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
punc_words=set([';',',','!','.',':'])

with open('train.csv', 'r',encoding='latin-1') as file: 

    data = csv.reader(file) 
  
    for row in data: 
        pol=row[0]
        if pol=='2':
            continue
        no=row[1]
        dte=row[2]
        qry=row[3]
        usr=row[4]
        text=row[5]
#        text = text.lower()
#        text = tokenizer.tokenize(text)
#        text = [ token for token in text if not token in stop_words ]
#        text = [ p_stemmer.stem(token) for token in text ]
        i=0
        lword=''
        if pol=='0':
            n1=n1+1
            for rword in re.split(r'[;,!.\s]\s*', text):
                if i==0:
#                    lword=rword
                    i=i+1
                    continue
                word=lword+rword
                if word in vocab:
                    vocab[word][0]=vocab[word][0]+1
                elif word!='': #and word[0]!='@':  
                    vocab[word]=[2,1]
                lword=rword    
#                else:
#                    print(word)
        else:
            n2=n2+1
            for rword in re.split(r'[;,!.\s]\s*', text):
                if i==0:
#                    lword=rword
                    i=i+1
                    continue
                word=lword+rword
                if word in vocab:
                    vocab[word][1]=vocab[word][1]+1
                elif word!='':  
                    vocab[word]=[1,2]
                lword=rword
#                else:
#                    print(word)               
            
#print(vocab)
n=n1+n2

ex=0
correct1=0
correct2=0
n1_test=0
n2_test=0


with open('test.csv', 'r') as file1: 
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
#        text = text.lower()
#        text = tokenizer.tokenize(text)
#        text = [ token for token in text if not token in stop_words ]
#        text = [ p_stemmer.stem(token) for token in text ]
        p1=math.log(n1/n)
        p2=math.log(n2/n)
        i=0
        lword=''
        for rword in re.split(r'[;,!.\s]\s*', text):
            if i==0:
                i=i+1
#                lword=rword 
                continue
            word=lword+rword
            if word in vocab:
                p1=p1+math.log(vocab[word][0])-math.log(vocab[word][0]+vocab[word][1])
                p2=p2+math.log(vocab[word][1])-math.log(vocab[word][0]+vocab[word][1])
            lword=rword        
#        print(p1)
#        print(p2) 
#        print(pol)
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
print(max(n1,n2)*100/(n))

print('confusion matrix:')
print('[[',correct1,n2_test-correct2,']')
print('[',n1_test-correct1,correct2,']]')

#accuracy:
#79.66573816155989
#random accuracy:
#50.0
#majority accuracy:
#50.69637883008357
#confusion matrix:
#actual
#141 37
#36 145
 
#without stemming
#accuracy:
#83.56545961002786
#random accuracy:
#50.0
#majority accuracy:
#50.69637883008357
#confusion matrix:
#actual
#157 39
#20 143



