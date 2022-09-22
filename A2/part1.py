import os
import glob
import argparse
import numpy as np

alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

def get_vocab(neg_txt_files, pos_txt_files):        
    vocab = set({})
    fmap = {}
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
                    vocab.add(w)
                    if w not in fmap:
                        fmap[w] = 0
                    fmap[w] += 1
        return
    make_vocab(pos_txt_files)
    make_vocab(neg_txt_files)
    return vocab, fmap

def train(path_neg, path_pos):        
    pos_txt_files = [os.path.normpath(i) for i in glob.glob(path_pos)]
    neg_txt_files = [os.path.normpath(i) for i in glob.glob(path_neg)]    

    vocab, fmap = get_vocab(neg_txt_files, pos_txt_files)
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

vocab, index, fmap, phi_y_0, phi_y_1, phi_y = train("part1_data/part1_data/train/neg/*.txt", "part1_data/part1_data/train/pos/*.txt")
print(max(fmap.values()))
print(min(fmap.values()))
print(max(phi_y_1))
print(min(phi_y_1))
print(max(phi_y_0))
print(min(phi_y_0))

def predict(text):
    text = text.split(' ')
    num = 1.0
    den1, den0 = 1.0, 1.0
    for word in text:
        w = ""
        for a in word.lower():
            if a in alphabet:
                w += a
            else:
                break
        num *= phi_y_1[index[w]]
        den1 *= phi_y_1[index[w]]
        den0 *= phi_y_0[index[w]]        
    num *= phi_y
    den = den1*phi_y + den0*(1-phi_y)
    prob = num/den
    return 0 if prob <= 0.5 else 1
