import numpy as np
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Bidirectional, RepeatVector
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras_fake_news_detector.library.utility.glove_loader import GLOVE_EMBEDDING_SIZE, load_glove

LATENT_SIZE = 256
BATCH_SIZE = 64
EPOCHS = 10
DOC2VEC_MAX_SEQ_LENGTH = 80
DOC2VEC_MAX_VOCAB_SIZE = 2000


def sentence_generator(X, embeddings, batch_size):
    while True:
        # loop once per epoch
        num_recs = X.shape[0]
        indices = np.random.permutation(np.arange(num_recs))
        num_batches = num_recs // batch_size
        for bid in range(num_batches):
            sids = indices[bid * batch_size: (bid + 1) * batch_size]
            Xbatch = embeddings[X[sids, :]]
            yield Xbatch, Xbatch


class Doc2Vec(object):
    model_name = 'doc2vec'

    def __init__(self, config, target_seq_length=None):
        if target_seq_length is None:
            target_seq_length = GLOVE_EMBEDDING_SIZE
        self.num_input_tokens = config['num_input_tokens']
        self.word2idx = config['word2idx']
        self.idx2word = config['idx2word']
        self.max_input_seq_length = config['max_input_seq_length']
        self.target_seq_length = target_seq_length
        self.config = config

        model = Sequential()
        model.add(Bidirectional(LSTM(LATENT_SIZE), input_shape=(self.max_input_seq_length, GLOVE_EMBEDDING_SIZE)))
        model.add(RepeatVector(self.max_input_seq_length))
        model.add(Bidirectional(LSTM(self.target_seq_length, return_sequences=True), merge_mode="sum"))
        model.compile(optimizer="sgd", loss="mse")
        self.model = model

        self.embedding = np.zeros((len(self.word2idx), GLOVE_EMBEDDING_SIZE))

    def load_glove(self, data_dir_path):
        word2em = load_glove(data_dir_path)

        unk_embed = np.random.uniform(-1, 1, GLOVE_EMBEDDING_SIZE)
        embedding = np.zeros((len(self.word2idx), GLOVE_EMBEDDING_SIZE))
        for word, idx in self.word2idx.items():
            vec = unk_embed
            if word in word2em:
                vec = word2em[word]
            embedding[idx] = vec

        embedding[self.word2idx["PAD"]] = np.zeros(shape=GLOVE_EMBEDDING_SIZE)
        embedding[self.word2idx["UNK"]] = unk_embed
        self.embedding = embedding

    def load_weights(self, weight_file_path):
        self.model.load_weights(weight_file_path)

    def transform_input_text(self, texts):
        temp = []
        for line in texts:
            x = []
            for word in line.lower().split(' '):
                wid = 1
                if word in self.word2idx:
                    wid = self.word2idx[word]
                x.append(wid)
                if len(x) >= self.max_input_seq_length:
                    break
            temp.append(x)
        temp = pad_sequences(temp, maxlen=self.max_input_seq_length)

        return temp

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + Doc2Vec.model_name + '-config.npy'

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + Doc2Vec.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + Doc2Vec.model_name + '-architecture.json'

    def fit(self, Xtrain, Xtest, epochs=None, model_dir_path=None):
        if epochs is None:
            epochs = EPOCHS
        if model_dir_path is None:
            model_dir_path = './models'

        config_file_path = Doc2Vec.get_config_file_path(model_dir_path)
        weight_file_path = Doc2Vec.get_weight_file_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        np.save(config_file_path, self.config)
        architecture_file_path = Doc2Vec.get_architecture_file_path(model_dir_path)
        open(architecture_file_path, 'w').write(self.model.to_json())

        Xtrain = self.transform_input_text(Xtrain)
        Xtest = self.transform_input_text(Xtest)

        train_gen = sentence_generator(Xtrain, self.embedding, BATCH_SIZE)
        test_gen = sentence_generator(Xtest, self.embedding, BATCH_SIZE)

        num_train_steps = len(Xtrain) // BATCH_SIZE
        num_test_steps = len(Xtest) // BATCH_SIZE

        history = self.model.fit_generator(train_gen,
                                           steps_per_epoch=num_train_steps,
                                           epochs=epochs,
                                           validation_data=test_gen,
                                           validation_steps=num_test_steps,
                                           callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)
        return history

    def predict(self, x):
        is_str = False
        if type(x) is str:
            is_str = True
            x = [x]

        Xtest = self.transform_input_text(x)
        Xtest = self.embedding[Xtest]

        preds = self.model.predict(Xtest)
        if is_str:
            preds = preds.flatten()
            return preds
        else:
            result = []
            for line in preds:
                result.append(line.flatten())
            return result

    def get_doc_vec_length(self):
        return self.max_input_seq_length * self.target_seq_length
