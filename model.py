from keras.models import Input, Model
from keras.layers import Embedding, Bidirectional, LSTM, Dense, GlobalAveragePooling1D
from args import Args
from spektral.layers import GCNConv


class Models:

    def bilstm_gat(self, matrix, id2token):

        inputs = Input(shape=(Args.maxlen,), dtype='int32')
        emb = Embedding(input_dim=len(id2token) + 1, output_dim=Args.emb_size,
                        weights=[matrix], mask_zero=False)(inputs)

        enc = Bidirectional(LSTM(units=Args.units, return_sequences=True,
                                 dropout=Args.dropout))(emb)

        graph = Input(shape=(Args.maxlen, Args.maxlen), dtype='int32')
        feats = GCNConv(channels=200)([enc, graph])
        feats = GlobalAveragePooling1D()(feats)
        outputs = Dense(2, activation='softmax')(feats)

        model = Model(inputs=[inputs, graph], outputs=outputs, name='bilstm_gcn')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def bilstm_gat_(self, id2token):

        inputs = Input(shape=(Args.maxlen,), dtype='int32')
        emb = Embedding(input_dim=len(id2token) + 1, output_dim=Args.emb_size,
                        mask_zero=False)(inputs)

        enc = Bidirectional(LSTM(units=Args.units, return_sequences=True,
                                 dropout=Args.dropout))(emb)

        graph = Input(shape=(Args.maxlen, Args.maxlen), dtype='int32')
        feats = GCNConv(channels=200)([enc, graph])
        feats = GlobalAveragePooling1D()(feats)
        outputs = Dense(2, activation='softmax')(feats)

        model = Model(inputs=[inputs, graph], outputs=outputs, name='bilstm_gcn')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model