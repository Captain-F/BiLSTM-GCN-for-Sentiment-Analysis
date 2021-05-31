import joblib


def read_pickle(path):
    with open(path, 'rb')as f:
        feats = joblib.load(f)

    return feats

def save_pickle(path, feats):
    with open(path, 'wb')as f:
        joblib.dump(feats, f)


def trans(labels, label2id):

    id2label = dict([j, i] for i, j in label2id.items())

    labels_trans = []
    for label in labels:
        seq_lab = []
        for i in label:
            lab = id2label[i]
            seq_lab.append(lab)
        labels_trans.append(seq_lab)
    return labels_trans


