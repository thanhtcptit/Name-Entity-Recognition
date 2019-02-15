import numpy as np
import tensorflow as tf
import pickle as pkl

def fix_data(file):
    with open(file, 'r') as f:
        data = f.read().split('\n')
    with open(file, 'w') as f:
        nextline = False
        for line in data:
            if line.find('-DOCSTART-') >= 0:
                nextline=True
                continue
            if list in ['', '\r']:
                if nextline:
                    continue
                f.write('\n')
                nextline = True
            else:
                f.write(line + '\n')
                nextline = False

def max_sentence_length(file):
    sent_len = []
    max_len = 0
    count = 0
    with open(file, 'r') as f:
        for line in f:
            if line in ['\n', '\r\n']:
                if count == 0:
                    continue
                if count > max_len:
                    max_len = count
                count = 0
            else:
                count += 1
    print ('Average sentence len: ', sum(sent_len) / len(sent_len))
    return max_len

def load_embedding_model():
    model = {}
    with open(glove_file, 'r') as f:
        for line in f:
            tokens = line.replace('\n', '').split(' ')
            model[tokens[0]] = np.array(tokens[1:], dtype=np.float32).tolist()
    return model

def one_hot_pos(tag):
    vector = [0] * 5
    if tag == 'NNP' or tag == 'NNPS':
        vector[0] = 1
    elif tag == 'NN' or tag == 'NNS':
        vector[1] = 1
    elif 'VB' in tag:
        vector[2] = 1
    elif tag == 'FW':
        vector[3] = 1
    else:
        vector[4] = 1
    return vector

def one_hot_chunk(tag):
    vector = [0] * 5
    if 'NP' in tag:
        vector[0] = 1
    elif 'VP' in tag:
        vector[1] = 1
    elif 'PP' in tag:
        vector[2] = 1
    elif 'O' in tag:
        vector[3] = 1
    else:
        vector[4] = 1
    return vector

def capital_word(word):
    if 65 <= ord(word[0]) <= 90:
        return [1]
    return [0]

def one_hot_ner(tag):
    vector = [0] * 9
    if tag == 'B-ORG':
        vector[0] = 1
    elif tag == 'I-ORG':
        vector[1] = 1
    elif tag == 'B-LOC':
        vector[2] = 1
    elif tag == 'I-LOC':
        vector[3] = 1
    elif tag == 'B-PER':
        vector[4] = 1
    elif tag == 'I-PER':
        vector[5] = 1
    elif tag == 'B-MISC':
        vector[6] = 1
    elif tag == 'I-MISC':
        vector[7] = 1
    else:
        vector[8] = 1
    return vector

def one_hot_ner1(tag):
    vector = [0] * 5
    if  'ORG' in tag:
        vector[0] = 1
    elif 'LOG' in tag:
        vector[1] = 1
    elif 'PER' in tag:
        vector[2] = 1
    elif 'MISC' in tag:
        vector[3] = 1
    else:
        vector[4] = 1
    return vector

def features_transform(file):
    model = load_embedding_model()
    sentences = []
    sentences_tag = []
    sentence = []
    sentence_tag = []
    with open(file, 'r') as f:
        for line in f:
            if line in ['\n', '\r\n']:
                if len(sentence) == 0:
                    continue
                if len(sentence) > max_sentence_len:
                    sentence = []
                    sentence_tag = []
                    continue

                padding = max_sentence_len - len(sentence)
                for i in range(padding):
                    sentence.append([0] * (word_dim + 11))
                    sentence_tag.append([0] * 5)
                sentences.append(sentence)
                sentences_tag.append(sentence_tag)
                sentence = []
                sentence_tag = []
            else:
                tokens = line.split(' ')
                assert len(tokens) == 4
                if tokens[0] in model:
                    word_emb = model[tokens[0]]
                elif tokens[0].lower() in model:
                    word_emb = model[tokens[0].lower()]
                else:
                    word_emb = np.random.uniform(size=(word_dim)).tolist()
                word = word_emb + one_hot_pos(tokens[1]) + \
                       one_hot_chunk(tokens[2]) + capital_word(tokens[0])
                word_tag = one_hot_ner1(tokens[3].strip())
                sentence.append(word)
                sentence_tag.append(word_tag)
    print ('Total sentences: ', len(sentences_tag))
    print ('Max sentence length: ', max_sentence_len)
    #pkl.dump(sentences, open('data/x_train_' + \
    #    str(max_sentence_len) + '.pkl', 'wb'))
    pkl.dump(sentences_tag[:10], open('data/sample/y_train_' + \
        str(max_sentence_len) + '.pkl', 'wb'))

def build_vocab(data):
    with open(data, 'r') as f:
        data = f.read().replace('\n', '').replace(' ', '')
    char = list(set(data)) + ['<']
    with open('data/cvocab.txt', 'w') as f:
        for i, c in enumerate(char):
            f.write(c + ' ' + str(i) + '\n')

def build_char_vocab(file):
    with open(file, 'r') as f:
        data = f.read()
    vocab = []
    for char in data:
        if char in vocab or char in ['\n', ' ']:
            continue
        vocab.append(char)
    with open('data/sample/char-vocab.txt', 'w') as f:
        f.write('\n'.join(vocab))

def max_word_len(file):
    max_len = 0
    with open(file, 'r') as f:
        for line in f:
            if line in ['\n', '\r\n']:
                continue
            else:
                tokens = line.split(' ')
                if (len(tokens[0]) > max_len):
                    max_len = len(tokens[0])
                    print (tokens[0])
    return max_len

def features_transform_charCNN(file):
    words = []
    words_tag = []
    with open(file, 'r') as f:
        for line in f:
            if line in ['\n', '\r\n']:
                continue
            else:
                tokens = line.split(' ')
                word_tag = one_hot_ner(tokens[3].strip())
                words.append(tokens[0])
                words_tag.append(word_tag)
    print ('Total words: ', len(words_tag))
    pkl.dump(words[:10], open('data/sample/x_train.pkl', 'wb'))
    pkl.dump(words_tag[:10], open('data/sample/y_train.pkl', 'wb'))

max_sentence_len = 10
max_word_length = 27
word_dim = 100
glove_file = "data/glove.6B.100d.txt"
features_transform_charCNN('data/preprocessed/train.txt')


