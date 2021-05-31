import os
from pyltp import Parser, Postagger
import pickle
import pandas as pd
import jieba
import numpy as np

csv = pd.read_csv(r'senti.csv')
contents = list(csv.content)
contents = [list(jieba.cut(c)) for c in contents]


# maxlen = 70

def build_matrix(dp, maxlen):
    matrix = np.zeros((maxlen, maxlen), dtype=np.int32)
    for i in range(maxlen):
        matrix[i][i] = 1
    for idx, i in enumerate(dp):
        print(idx)
        if i != 0:
            matrix[i - 1][idx] = 1
            matrix[idx][i - 1] = 1
    return matrix


def dump_file(path, data):
    with open(path, 'wb')as f:
        pickle.dump(data, f)


with open(r'stopwords.txt', 'r', encoding='utf8')as f:
    sw = f.readlines()

sw = [s.strip() for s in sw]
sents = []
for c in contents:
    sent = []
    for w in c:
        if w not in sw and w != ' ':
            sent.append(w)
    sents.append(sent)

parser = Parser()
parser.load(r'ltp/parser.model')
postagger = Postagger()
postagger.load(r'ltp/pos.model')

dp_matrix = []
for sent in sents:
    postags = postagger.postag(sent)
    arcs = parser.parse(sent, postags)
    arc_head = [arc.head for arc in arcs]
    matrix = build_matrix(arc_head, 70)
    dp_matrix.append(matrix)

dump_file(r'dp_graph.pickle', dp_matrix)