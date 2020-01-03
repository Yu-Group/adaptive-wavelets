from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras_fake_news_detector.library.utility.glove_loader import GLOVE_EMBEDDING_SIZE, load_glove
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras_fake_news_detector.library.encoders.doc2vec import Doc2Vec
import numpy as np

BATCH_SIZE = 64
VERBOSE = 1
EPOCHS = 50
MAX_SEQ_LENGTH = 2000


def generate_batch(x_samples, y_samples):
    num_batches = len(x_samples) // BATCH_SIZE

    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * BATCH_SIZE
            end = (batchIdx + 1) * BATCH_SIZE
            yield x_samples[start:end], y_samples[start:end]


class GloveFeedforwardNet(object):
    model_name = 'glove-feed-forward'

    def __init__(self, config):
        self.num_target_tokens = config['num_target_tokens']
        self.config = config

        model = Sequential()
        model.add(Dense(units=64, input_dim=GLOVE_EMBEDDING_SIZE, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.num_target_tokens, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model

        self.word2em = dict()

    def load_glove(self, data_dir_path):
        self.word2em = load_glove(data_dir_path)

    def load_weights(self, weight_file_path):
        self.model.load_weights(weight_file_path)

    def transform_input_text(self, texts):
        record_count = len(texts)
        X = np.zeros(shape=(record_count, GLOVE_EMBEDDING_SIZE))
        print('records: ', record_count)
        for i, line in enumerate(texts):
            text = line.lower().split(' ')
            seq_length = min(len(text), MAX_SEQ_LENGTH)
            E = np.zeros(shape=(GLOVE_EMBEDDING_SIZE, seq_length))
            for j in range(seq_length):
                word = text[j]
                if word in self.word2em:
                    E[:, j] = self.word2em[word]
            X[i, :] = np.sum(E, axis=1)
        return X

    def transform_target_encoding(self, targets):
        return np_utils.to_categorical(targets, num_classes=self.num_target_tokens)

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, epochs=None, model_dir_path=None):
        if epochs is None:
            epochs = EPOCHS
        if model_dir_path is None:
            model_dir_path = './models'

        config_file_path = model_dir_path + '/' + self.model_name + '-config.npy'
        weight_file_path = model_dir_path + '/' + self.model_name + '-weights.h5'
        checkpoint = ModelCheckpoint(weight_file_path)
        np.save(config_file_path, self.config)
        architecture_file_path = model_dir_path + '/' + self.model_name + '-architecture.json'
        open(architecture_file_path, 'w').write(self.model.to_json())

        Ytrain = self.transform_target_encoding(Ytrain)
        Ytest = self.transform_target_encoding(Ytest)

        Xtrain = self.transform_input_text(Xtrain)
        Xtest = self.transform_input_text(Xtest)

        train_gen = generate_batch(Xtrain, Ytrain)
        test_gen = generate_batch(Xtest, Ytest)

        train_num_batches = len(Xtrain) // BATCH_SIZE
        test_num_batches = len(Xtest) // BATCH_SIZE

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=VERBOSE, validation_data=test_gen, validation_steps=test_num_batches,
                                           callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)
        return history

    def predict(self, x):
        is_str = False
        if type(x) is str:
            is_str = True
            x = [x]

        Xtest = self.transform_input_text(x)

        preds = self.model.predict(Xtest)
        if is_str:
            preds = preds[0]
            return np.argmax(preds)
        else:
            return np.argmax(preds, axis=1)


class Doc2VecFeedforwardNet(object):
    model_name = 'doc2vec-feed-forward'

    def __init__(self, config):
        self.num_target_tokens = config['num_target_tokens']
        self.config = config
        self.doc2vec = Doc2Vec(config)

        model = Sequential()
        model.add(Dense(units=64, input_dim=self.doc2vec.get_doc_vec_length(), activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.num_target_tokens, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def load_glove(self, data_dir_path):
        self.doc2vec.load_glove(data_dir_path)

    def load_weights(self, weight_file_path):
        self.model.load_weights(weight_file_path)

    def transform_input_text(self, texts):
        record_count = len(texts)
        X = np.zeros(shape=(record_count, self.doc2vec.get_doc_vec_length()))
        print('records: ', record_count)
        for i, line in enumerate(texts):
            X[i, :] = self.doc2vec.predict(line)
        return X

    def transform_target_encoding(self, targets):
        return np_utils.to_categorical(targets, num_classes=self.num_target_tokens)

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, epochs=None, model_dir_path=None):
        if epochs is None:
            epochs = EPOCHS
        if model_dir_path is None:
            model_dir_path = './models'

        config_file_path = model_dir_path + '/' + self.model_name + '-config.npy'
        weight_file_path = model_dir_path + '/' + self.model_name + '-weights.h5'
        checkpoint = ModelCheckpoint(weight_file_path)
        np.save(config_file_path, self.config)
        architecture_file_path = model_dir_path + '/' + self.model_name + '-architecture.json'
        open(architecture_file_path, 'w').write(self.model.to_json())

        Ytrain = self.transform_target_encoding(Ytrain)
        Ytest = self.transform_target_encoding(Ytest)

        Xtrain = self.transform_input_text(Xtrain)
        Xtest = self.transform_input_text(Xtest)

        train_gen = generate_batch(Xtrain, Ytrain)
        test_gen = generate_batch(Xtest, Ytest)

        train_num_batches = len(Xtrain) // BATCH_SIZE
        test_num_batches = len(Xtest) // BATCH_SIZE

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=VERBOSE, validation_data=test_gen, validation_steps=test_num_batches,
                                           callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)
        return history

    def predict(self, x):
        is_str = False
        if type(x) is str:
            is_str = True
            x = [x]

        Xtest = self.transform_input_text(x)

        preds = self.model.predict(Xtest)
        if is_str:
            preds = preds[0]
            return np.argmax(preds)
        else:
            return np.argmax(preds, axis=1)