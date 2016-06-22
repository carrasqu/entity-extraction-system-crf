from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
import numpy as np
import sys

def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.
    
    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_) #- {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )


def word2features(sent, i):
    word = sent[i][0]
    features = [
        'bias',
        'word=' + word,
        'word.isdigit=%s' % word.isdigit()
    ] 

    if i > 0:
        word1 = sent[i-1][0]
        features.extend([
            '-1:word=' + word1,
            '-1:word.isdigit=%s' % word1.isdigit()   
        ])
    else:
        features.append('BOS')
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.extend([
            '+1:word=' + word1,
            '+1:word.isdigit=%s' % word1.isdigit()
        ])
    else:
        features.append('EOS')
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]    

def readdata(fname='news_tagged_data.txt',percentage_train=80):
      i=0
      z=0  
      with open(fname) as f:
        for line in f:
         if i==0:
            tset=[(line.split()[0],line.split()[1])]
            i+=1
         elif (line.isspace() == False)&(i>0):
            tset.append( (line.split()[0],line.split()[1]) )
            i+=1
         elif (line.isspace() == True):
            #sys.exit()
            if z==0 :
             alldata=[tset]
             z=1
             i=0
             #break
            else:
             alldata.append(tset)
             i=0
      itrain=int(percentage_train*len(alldata)/100)
      return alldata[0:itrain],alldata[itrain:] #,alldata



# do stuff here
# read the data
train_sents,test_sents=readdata()


X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]



# training the model
trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)


trainer.set_params({
    'c1': 0.5,   # coefficient for L1 penalty
    'c2': 1e-4,  # coefficient for L2 penalty
    'max_iterations': 25000,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})


trainer.train('NER_maluuba')

tagger = pycrfsuite.Tagger()
tagger.open('NER_maluuba')


example_sent = [test_sents[0][0]]
print(' '.join(sent2tokens(example_sent)))

print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
print("Correct:  ", ' '.join(sent2labels(example_sent)))

y_pred = [tagger.tag(xseq) for xseq in X_test]

print(bio_classification_report(y_test, y_pred))
