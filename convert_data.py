import numpy as np
import time
import pickle as pkl

def check_tags(word):
    tmp = word.rsplit('/', 2)
    if len(tmp) > 2:
        word = '/'.join(tmp[:-1]).replace('\\', '')
        tag = tmp[-1].split('|')[0]
    else:
        word = tmp[0]
        tag = tmp[-1].split('|')[0]
    return word, tag

def one_hot_pos(tag):
    vector = [0] * len(vocab)
    vector[vocab.index(tag)] = 1
    return vector

def features_tranform(mode):
    words = []
    tags = []
    with open('data/pos/' + mode + '.txt') as f:
        for _ in f:
            line = _.rstrip('\n')
            tokens = line.split(' ')[:-1]
            for w in tokens:
                if '-LRB-' in w or '-RRB-' in w:
                    continue
                word, tag = check_tags(w)
                words.append(word)
                tags.append(one_hot_pos(tag))

    assert len(words) == len(tags)
    pkl.dump(words[:10], open('data/pos/' + mode + '_word.pkl', 'wb'))
    pkl.dump(tags[:10], open('data/pos/' + mode + '_tag.pkl', 'wb'))


def load_vocab():
    with open('data/pos/vocab.txt') as f:
        vocab = f.read().split('\n')
    return vocab


def load_data(data_file_path):
    dictionary = {}
    with open(data_file_path) as f:
        for line in f:
            elements = line.split(' ')
            word = str(elements[0])
            word_vector = elements[1:]
            word_vector = [float(e) for e in word_vector]
            word_vector = np.asarray(word_vector, dtype=np.float32)
            dictionary[word] = word_vector
    return dictionary

def build_vocab():
    tags = []
    with open('data/pos/train.txt') as f:
        for _ in f:
            line = _.rstrip('\n')
            tokens = line.split(' ')[:-1]
            for w in tokens:
                if '-LRB-' in w or '-RRB-' in w:
                    continue
                word, tag = check_tags(w)
                if tag in tags:
                    continue
                tags.append(tag)
    with open('data/pos/vocab.txt', 'w') as f:
        f.write('\n'.join(tags))

def build_char_vocab():
    with open('data/pos/train.txt', 'r') as f:
        data = f.read()
    with open('data/pos/dev.txt', 'r') as f:
        data += ' ' + f.read()
    with open('data/pos/test.txt', 'r') as f:
        data += ' ' + f.read()
    char_vocab = []
    for char in data:
        if char in char_vocab or char in ['\n', '\r\n']:
            continue
        char_vocab.append(char)
    with open('data/pos/char-vocab.txt', 'w') as f:
        f.write('\n'.join(char_vocab))

def max_word_len():
    max_len = 0
    with open('data/pos/train.txt', 'r') as f:
        for _ in f:
            line = _.rstrip('\n')
            tokens = line.split(' ')[:-1]
            for w in tokens:
                if '-LRB-' in w or '-RRB-' in w:
                    continue
                word, tag = check_tags(w)
                if len(word) > max_len:
                    max_len = len(word)
                    print (word)
    print (max_len)

if __name__ == '__main__':
    start_time = time.time()
    vocab = load_vocab()
    #features_tranform('train')
    max_word_len()

    print("DONE: --- %s seconds ---" % (time.time() - start_time))