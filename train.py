import logging
from args import Args
from model import Models
from load_data import LoadData
from utils import *
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO, format=('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.info('加载数据中...')
    sents, labels = LoadData().to_vec()
    logger.info('sents.shape = {}, labels.shape = {}'.format(sents.shape, labels.shape))
    graphs = read_pickle(r'dp_graph.pickle')
    graphs = np.array(graphs)
    logger.info('加载文本图...')
    logger.info('graphs.shape = {}'.format(graphs.shape))
    id2word = LoadData().build_corpus()
    logger.info('加载词向量...')
    matrix, num = LoadData().embedding()
    logger.info('共有{}个未登录词...'.format(num))
    trainX, testX, gTrainX, gTestX, trainY, testY = train_test_split(sents, graphs, labels, test_size=0.2)
    logger.info('trainX.shape = {}, testX.shape = {}'.format(trainX.shape, testX.shape))
    myModel = Models().bilstm_gat(matrix=matrix, id2token=id2word)
    myModel.fit([trainX, gTrainX], trainY,
                validation_data=([testX, gTestX], testY),
                batch_size=Args.batch_size,
                epochs=Args.epochs, verbose=1)
    pre = myModel.predict([testX, gTestX], batch_size=Args.batch_size, verbose=1)
    pre_ = pre.argmax(axis=-1)
    testY_ = testY.argmax(axis=-1)
    report = classification_report(testY_, pre_, digits=5)
    print(report)


