import os
import glob
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',"'"]
stopwords = stopwords.words('english')

def get_vocab(neg_txt_files, pos_txt_files, remove_stopwords=False, stemming=False):        
    vocab = set({})
    fmap = {}
    if stemming:
        ps = PorterStemmer()
    def make_vocab(source):
        for text_files in source:            
            with open(text_files, 'r', encoding='utf8') as file:          
                content = file.read()
                words = content.split(' ')
                for word in words:         
                    w = ""
                    for a in word.lower():
                        if a in alphabet:
                            w += a
                        else:
                            break
                    if remove_stopwords and w in stopwords:
                        continue                    
                    if stemming:
                        w = ps.stem(w)
                    vocab.add(w)
                    if w not in fmap:
                        fmap[w] = 0
                    fmap[w] += 1
        return
    make_vocab(pos_txt_files)
    make_vocab(neg_txt_files)
    return vocab, fmap

def train(path_neg, path_pos, remove_stopwords=False, stemming=False):        
    pos_txt_files = [os.path.normpath(i) for i in glob.glob(path_pos)]
    neg_txt_files = [os.path.normpath(i) for i in glob.glob(path_neg)]    

    if stemming:
        ps = PorterStemmer()
    vocab, fmap = get_vocab(neg_txt_files, pos_txt_files, remove_stopwords, stemming)
    vocab = sorted(vocab)    
    index = {}
    for i, word in enumerate(vocab):
        index[word] = i                

    m = len(pos_txt_files) + len(neg_txt_files)
    def feature_vector(text, dims):
        text = text.split(' ')
        x = np.zeros((dims,1))
        for word in text:    
            w = ""
            for a in word.lower():
                if a in alphabet:
                    w += a
                else:
                    break
            if remove_stopwords and w in stopwords:
                continue
            if stemming:
                w = ps.stem(w)
            x[index[w]] = 1
        return x

    def find_phi(files):
        phi = np.zeros((len(vocab), 1))
        for file in files:
            with open(file, 'r', encoding='utf8') as f:    
                phi += feature_vector(f.read(), len(vocab))
        phi += 1
        phi *= (1/(len(files)+2))
        return phi

    phi_y_1 = find_phi(pos_txt_files)
    phi_y_0 = find_phi(neg_txt_files)
    phi_y = len(pos_txt_files)/m
    return vocab, index, fmap, phi_y_0, phi_y_1, phi_y

vocab, index, fmap, phi_y_0, phi_y_1, phi_y = train("part1_data/part1_data/train/neg/*.txt", "part1_data/part1_data/train/pos/*.txt", remove_stopwords=True, stemming=True)
#print(max(fmap.values()))
#print(min(fmap.values()))
#print(max(phi_y_1))
#print(min(phi_y_1))
#print(max(phi_y_0))
#print(min(phi_y_0))

def predict(text, remove_stopwords=False, stemming=False):
    if stemming:
        ps = PorterStemmer()
    text = text.split(' ')
    prob = 0.0
    num = 0.0
    den0, den1 = 1.0, 1.0
    for word in text:
        w = ""
        for a in word:
            if a not in alphabet:
                break
            w += a        
        if remove_stopwords and w in stopwords:
            continue
        if stemming:
            w = ps.stem(w)
        if w in index:           
            num += np.log(phi_y_1[index[w]])
            den0 *= phi_y_0[index[w]]
            den1 *= phi_y_1[index[w]]
        else:
            den0 *= 7.9987202e-05
            den1 *= 7.9987202e-05
            num += np.log(7.9987202e-05)
    num += np.log(phi_y)
    den = np.log(den1*phi_y + den0*(1-phi_y))
    prob = num - den
    return 0 if prob <= np.log(0.5) else 1

def accuracy(path_neg, path_pos, remove_stopwords=False, stemming=False):
    pos_txt_files = [os.path.normpath(i) for i in glob.glob(path_pos)]
    neg_txt_files = [os.path.normpath(i) for i in glob.glob(path_neg)]
    
    correct_positive, correct_negative = 0, 0    
    def correct(files, k):
        ans = 0
        for txt_files in files:
            with open(txt_files, 'r', encoding='utf8') as file:
                text = file.read()
                if predict(text, remove_stopwords, stemming) == k:
                    ans += 1
        return ans

    correct_positive += correct(pos_txt_files,1)
    correct_negative += correct(neg_txt_files,0)

    return (correct_negative + correct_positive) / (len(pos_txt_files) + len(neg_txt_files))

def word_cloud(path_neg, path_pos):
    pos_txt_files = [os.path.normpath(i) for i in glob.glob(path_pos)]
    neg_txt_files = [os.path.normpath(i) for i in glob.glob(path_neg)]  
    
    def make_word_cloud(text, name):
        stopwords = set()
        stopwords.add('br')
        wc = WordCloud(
            background_color='white',
            max_words=1000,
            stopwords=stopwords
        )
        wc.generate(text)   
        #fig = plt.figure()
        #fig.set_figwidth(14) # set width
        #fig.set_figheight(18) # set height
        plt.imshow(wc, interpolation='bilinear')
        plt.tight_layout(pad=0)
        plt.axis('off')
        plt.show()        
        return
    
    text = ''
    for txt_file in pos_txt_files:
        with open(txt_file, 'r', encoding='utf8') as file:
            text += file.read().strip()
            text += ' '
    make_word_cloud(text, 'positive')
    text = ''
    for txt_file in neg_txt_files:
        with open(txt_file, 'r', encoding='utf8') as file:
            text += file.read().strip()
            text += ' '
    make_word_cloud(text, 'negative')
    return

def random_prediction_accuracy(path_neg, path_pos):
    pos_txt_files = [os.path.normpath(i) for i in glob.glob(path_pos)]
    neg_txt_files = [os.path.normpath(i) for i in glob.glob(path_neg)]
    
    correct = 0
    def compute(n, target):
        correct = 0
        for i in range(n):
            pred = np.random.randint(0, 2)
            if pred == target:
                correct += 1
        return correct
    correct += compute(len(pos_txt_files), 1)
    correct += compute(len(neg_txt_files), 0)

    accuracy = correct / (len(pos_txt_files) + len(neg_txt_files))
    return accuracy

def all_positive_accuracy(path_neg, path_pos):
    pos_txt_files = [os.path.normpath(i) for i in glob.glob(path_pos)]
    neg_txt_files = [os.path.normpath(i) for i in glob.glob(path_neg)]

    correct = 0
    def compute(n, target):
        correct = 0
        for i in range(n):
            if target == 1:
                correct += 1
        return correct
    correct += compute(len(pos_txt_files), 1)
    correct += compute(len(neg_txt_files), 0)

    return correct / (len(pos_txt_files) + len(neg_txt_files))

def confusion_matrix_naive_bayes(path_neg, path_pos):
    pos_txt_files = [os.path.normpath(i) for i in glob.glob(path_pos)]
    neg_txt_files = [os.path.normpath(i) for i in glob.glob(path_neg)]
     
    def compute(files, target):
        true_targets, false_targets = 0, 0
        for txt_file in files:
            with open(txt_file, 'r', encoding='utf8') as file:
                text = file.read()
                if predict(text) == target:
                    true_targets += 1
                else:
                    false_targets += 1
        return true_targets, false_targets
    true_positives, false_positives = compute(pos_txt_files, 1)
    true_negatives, false_negatives = compute(neg_txt_files, 0)
    return true_positives, false_positives, true_negatives, false_negatives

def confusion_matrix_all_positive(path_neg, path_pos):
    pos_txt_files = [os.path.normpath(i) for i in glob.glob(path_pos)]
    neg_txt_files = [os.path.normpath(i) for i in glob.glob(path_neg)]

    true_positives, false_positives = len(pos_txt_files), 0
    true_negatives, false_negatives = 0, len(neg_txt_files)
    return true_positives, false_positives, true_negatives, false_negatives

def confusion_matrix_random_prediction(path_neg, path_pos):
    pos_txt_files = [os.path.normpath(i) for i in glob.glob(path_pos)]
    neg_txt_files = [os.path.normpath(i) for i in glob.glob(path_neg)]

    true_positives, false_positives = 0, 0
    for i in range(len(pos_txt_files)):
        pred = np.random.randint(0, 2)
        if pred == 1:
            true_positives += 1
        else:
            false_positives += 1
    true_negatives, false_negatives = 0, 0
    for i in range(len(neg_txt_files)):
        pred = np.random.randint(0, 2)
        if pred == 1:
            false_negatives += 1
        else:
            true_negatives += 1
    return true_positives, false_positives, true_negatives, false_negatives

#print(accuracy("part1_data/part1_data/train/neg/*.txt", "part1_data/part1_data/train/pos/*.txt")) #83.072%
#print(accuracy("part1_data/part1_data/test/neg/*.txt", "part1_data/part1_data/test/pos/*.txt")) #77.274%

#word_cloud("part1_data/part1_data/train/neg/*.txt", "part1_data/part1_data/train/pos/*.txt")
#word_cloud("part1_data/part1_data/test/neg/*.txt", "part1_data/part1_data/test/pos/*.txt")

#print(random_prediction_accuracy("part1_data/part1_data/test/neg/*.txt", "part1_data/part1_data/test/pos/*.txt")) #49.77%
#print(all_positive_accuracy("part1_data/part1_data/test/neg/*.txt", "part1_data/part1_data/test/pos/*.txt")) #66%

#print(confusion_matrix_naive_bayes("part1_data/part1_data/test/neg/*.txt", "part1_data/part1_data/test/pos/*.txt")) #(7529, 2471, 4099, 901) P = 0.7529, R = 0.8932, F1-score = 0.8170
#print(confusion_matrix_random_prediction("part1_data/part1_data/test/neg/*.txt", "part1_data/part1_data/test/pos/*.txt")) #(5025, 4975, 2457, 2543)
#print(confusion_matrix_all_positive("part1_data/part1_data/test/neg/*.txt", "part1_data/part1_data/test/pos/*.txt")) #(10000, 0, 0, 5000) 

#with stemming and stopword removal
#print(accuracy("part1_data/part1_data/train/neg/*.txt", "part1_data/part1_data/train/pos/*.txt", remove_stopwords=True, stemming=True)) #85.60%
#print(accuracy("part1_data/part1_data/test/neg/*.txt", "part1_data/part1_data/test/pos/*.txt", remove_stopwords=True, stemming=True)) #81.18%