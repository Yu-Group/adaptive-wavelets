from collections import Counter

MAX_INPUT_SEQ_LENGTH = 500
MAX_VOCAB_SIZE = 2000


def fit_input_text(X, max_input_seq_length=None, max_vocab_size=None):
    if max_input_seq_length is None:
        max_input_seq_length = MAX_INPUT_SEQ_LENGTH
    if max_vocab_size is None:
        max_vocab_size = MAX_VOCAB_SIZE

    input_counter = Counter()
    max_seq_length = 0
    for line in X:
        text = [word.lower() for word in line.split(' ')]
        seq_length = len(text)
        if seq_length > max_input_seq_length:
            text = text[0:max_input_seq_length]
            seq_length = len(text)
        for word in text:
            input_counter[word] += 1
        max_seq_length = max(max_seq_length, seq_length)

    word2idx = dict()

    for idx, word in enumerate(input_counter.most_common(max_vocab_size)):
        word2idx[word[0]] = idx + 2
    word2idx['PAD'] = 0
    word2idx['UNK'] = 1
    idx2word = dict([(idx, word) for word, idx in word2idx.items()])
    num_input_tokens = len(word2idx)
    config = dict()
    config['word2idx'] = word2idx
    config['idx2word'] = idx2word
    config['num_input_tokens'] = num_input_tokens
    config['max_input_seq_length'] = max_seq_length

    return config
