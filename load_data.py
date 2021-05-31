import pandas as pd
from utils import *
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import numpy as np
from args import Args

class LoadData:

    def __init__(self):
        self.maxlen = Args.maxlen
        self.sents = read_pickle(r'sents.pickle')
        self.labels = list(pd.read_csv(r'senti.csv').label)

    def build_corpus(self):

        raw_words = [w for s in self.sents for w in s]
        words = list(set(raw_words))
        words.sort(key=raw_words.index)
        id2word = {idx+1: i for idx, i in enumerate(words)}

        return id2word

    def to_vec(self):

        id2word = self.build_corpus()
        word2id = {token: idx for idx, token in id2word.items()}
        sents_id = []
        for s in self.sents:
            sent = []
            for w in s:
                sent.append(word2id[w])
            sents_id.append(sent)

        sents_pad = self.pad_seq(sents_id)
        labels_oneHot = np_utils.to_categorical(self.labels, 2)

        return sents_pad, labels_oneHot

    def embedding(self):

        id2word = self.build_corpus()
        word2id = {token: idx for idx, token in id2word.items()}

        w_ = []
        v_ = []
        num = 0

        matrix = np.zeros((len(id2word)+1, Args.emb_size))

        path = r'sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5(1)'
        with open(path, 'r', encoding='utf8')as f:
            w_v_arrs = f.readlines()

        for w_v in w_v_arrs:
            w = w_v.strip().split(' ')[0]
            v = [float(i) for i in w_v.strip().split(' ')[1:]]
            w_.append(w)
            v_.append(v)

        w_v_dict = dict([[w, v] for w, v in zip(w_, v_)])

        for word, idx in word2id.items():
            if word in w_v_dict.keys():
                matrix[idx] = w_v_dict[word]
            else:
                matrix[idx] = list(np.random.uniform(-0.25, 0.25, Args.emb_size))
                num = num + 1

        #print('%d个词不在词表中' % num)

        return matrix, num

    def pad_seq(self, x):

        xPad = pad_sequences(x, maxlen=self.maxlen, padding='post', truncating='post')

        return xPad

