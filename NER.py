from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
import numpy as np
import sys


def word2features(sent, i):
    word = sent[i][0]
    features = [
        'bias',
        'word=' + word,
        'word.isdigit=%s' % word.isdigit()
    ]

    if i > 0:
        wordm1 = sent[i-1][0]
        features.extend([
            '-1:word=' + wordm1,
            '-1:word.isdigit=%s' % wordm1.isdigit()
        ])
    else:
        features.append('BOS')

    if i < len(sent)-1:
        wordp1 = sent[i+1][0]
        features.extend([
            '+1:word=' + wordp1,
            '+1:word.isdigit=%s' % wordp1.isdigit()
        ])
    else:
        features.append('EOS')

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def pquery(s):
      ss=np.asarray(s.split())
      i=0
      z=0
      for line in ss:
         if i==0:
            tset=[(line.split()[0],'NAL')]
            i+=1
         elif (line.isspace() == False)&(i>0):
            tset.append( (line.split()[0], 'NAL') )
            i+=1
      return tset 



tagger = pycrfsuite.Tagger()
tagger.open('NER_maluuba') # loading the trained model


# reading from std input
while True :
  s=raw_input('Type a query (type "exit" to exit): \n')
  print ""
  if s=='exit':
     break
  elif s=='':
     print 'enter a valid query '
  else:
     example=pquery(s)
     prediction=tagger.tag(sent2features(example))
     z=0
     for i in s.split():
         print i, prediction[z] 
         z+=1
     print ' ' 
