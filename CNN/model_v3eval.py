from __future__ import print_function
import tensorflow as tf
from tensorflow.python.client import device_lib
import pickle as pkl
import numpy as np


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


print('GPU: ', get_available_gpus())
print('Tensorflow version: ', tf.__version__)
print('Numpy version: ', np.__version__)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def get_data(sentence_file, tag_file):
    with open(sentence_file, 'r') as f:
        sentences = f.read().split('\n')[:-1]
    tags = pkl.load(open(tag_file, 'rb'))
    return sentences, tags


def get_vocab(file):
    vocab = {}
    reversed_vocab = {}
    with open(file, 'r') as f:
        items = f.read().split('\n')
        for i, w in enumerate(items):
            if w.lower() in vocab:
                continue
            vocab[w.lower()] = i
            reversed_vocab[i] = w.lower()
    return vocab, reversed_vocab


def sentence_to_id(sentence, vocab, max_len):
    tokens = sentence.split(' ')
    sentence_len = len(tokens)
    pad_length = max_len - sentence_len
    if WINDOW_SIZE > 1:
        return [0] * 2 + [vocab[t] if t in vocab else vocab['<UNK>'] for t in tokens] \
               + [0] * (pad_length + 2), sentence_len
    else:
        return [vocab[t] if t in vocab else vocab['<UNK>'] for t in tokens] \
               + [0] * pad_length, sentence_len


def id_to_sentence(idx, reversed_vocab):
    return ' '.join([reversed_vocab[i] for i in idx])


def transform(sentence, vocab):
    x_transform = []
    for s in sentence:
        sent_idx, length = sentence_to_id(s, vocab, MAX_SENTENCE_LEN)
        x_transform.append(sent_idx)
    return x_transform

def load_params(model_dir):
    params = []
    with open(model_dir + '/params.txt', 'r') as f:
        for i, line in enumerate(f):
            if i == 2:
                params.append(np.array(line.split(' '), dtype=np.int32))
            else:
                params.append(int(line))
    return params

def load_embedding_model():
    pretrain_vocab = []
    pretrain_embedding = []
    with open("../data/glove.6B.100d.txt", 'r') as f:
        for line in f:
            tokens = line.replace('\n', '').split(' ')
            pretrain_vocab.append(tokens[0])
            pretrain_embedding.append(np.array(tokens[1:],
                dtype=np.float32).tolist())
    return pretrain_vocab, np.array(pretrain_embedding)

def embed_tensor(string_tensor, trainable=True):
    global vocab
    pretrain_vocab, pretrain_embedding = load_embedding_model()
    word_only_in_train = set(vocab) - set(pretrain_vocab)
    print ('Number of word only in train: ', len(word_only_in_train))
    train_vocab = pretrain_vocab + list(word_only_in_train)
    vocab_lookup = tf.contrib.lookup.index_table_from_tensor(
        mapping=tf.constant(train_vocab), default_value=len(train_vocab))
    string_tensor = vocab_lookup.lookup(string_tensor)
    pretrain_embedding = tf.get_variable(name="pretrain_embs",
        shape=pretrain_embedding.shape, dtype=tf.float32)
    train_embedding = tf.get_variable(name="embs_only_in_train",
        shape=[len(word_only_in_train), EMBEDDING_SIZE])
    unk_embedding = tf.get_variable(name="unk_embs", shape=[1, EMBEDDING_SIZE])
    embedding = tf.concat(
        [pretrain_embedding, train_embedding, unk_embedding], axis=0)
    return tf.nn.embedding_lookup(embedding, string_tensor)

def write_result(file, prediction):
    label = ['B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER',
             'B-MISC', 'I-MISC', 'O']
    words = []
    with open(file, 'r') as f:
        for line in f:
            if line in ['\n', '\r\n']:
                words.append(line)
                continue
            tokens = line.strip().split(' ')
            assert len(tokens) == 4
            words.append([tokens[0], tokens[1], tokens[3], 'x'])
    j = 0
    flag = False
    for i, p in enumerate(prediction):
        while words[j] in ['\n', '\r\n']:
            j += 1
            if j == len(words):
                flag = True
                break
        if flag:
            break
        words[j][3] = label[p]
        j += 1

    with open(model_dir +'/prediction.txt', 'w') as f:
        for word in words:
            f.write(' '.join(word))
            f.write('\n')

model_dir = 'model_4'

vocab, reversed_vocab = get_vocab('../data/vocab.txt')
sentences_test, tags_test = get_data('../data/sentences_test.txt',
                                     '../data/y_test_124.pkl')
# Hyper params
EMBEDDING_SIZE, WINDOW_SIZE, FILTER_SIZE, NUM_FILTER = load_params(model_dir)
TOTAL_FILTERS = NUM_FILTER * len(FILTER_SIZE)

VOCAB_SIZE = len(vocab)
TEST_SIZE = len(sentences_test)
MAX_SENTENCE_LEN = 124
NUM_CLASSES = 9

print('Test size: ', TEST_SIZE)

input_sentences = tf.placeholder(tf.string,
                    [None, WINDOW_SIZE],
                    'input_sentences')
output_tags = tf.placeholder(tf.int32,
                    [None, NUM_CLASSES],
                    'output_tags')

input_vector = embed_tensor(input_sentences)
input_vector = tf.expand_dims(input_vector, -1)

filter_weights = {
    'fw' + str(i + 1): tf.Variable(tf.random_normal(
        [FILTER_SIZE[i], EMBEDDING_SIZE, 1, NUM_FILTER], stddev=0.1),
        name='fw' + str(i + 1)) for i in range(len(FILTER_SIZE))
}

fc_weight = tf.get_variable(shape=[TOTAL_FILTERS, NUM_CLASSES],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            name='fc')

filter_bias = {
    'fb' + str(i + 1): tf.Variable(
        tf.constant(0.1, shape=[NUM_FILTER]), name='fb' + str(i + 1))
    for i in range(len(FILTER_SIZE))
}


def conv2d(x, W, b, strides=1):
    x1 = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1],
                      padding='VALID', name='conv2d')
    x1 = tf.nn.bias_add(x1, b)
    return tf.nn.relu(x1, name='relu')


def max_pool2d(x, k, stride=1):
    return tf.nn.max_pool(x, ksize=[1, k[0], k[1], 1],
                          strides=[1, stride, stride, 1],
                          padding='VALID', name='max_pool2d')


def apply_CNN(x):
    conv_output = []
    for i in range(len(FILTER_SIZE)):
        conv = conv2d(x, filter_weights['fw' + str(i + 1)],
                    filter_bias['fb' + str(i + 1)])
        conv_output.append(max_pool2d(conv,
                    [WINDOW_SIZE- FILTER_SIZE[i] + 1, 1]))
    conv_output = tf.reshape(tf.concat(conv_output, 3), [-1, TOTAL_FILTERS])
    fc = tf.matmul(conv_output, fc_weight)

    return fc


# Predict
prediction = tf.nn.softmax(apply_CNN(input_vector))
train_prediction = tf.argmax(prediction, 1, name="predictions")
correct_predictions = tf.equal(train_prediction,
                               tf.argmax(output_tags, 1))
accuracy = tf.reduce_mean(
    tf.cast(correct_predictions, "float"), name="accuracy")

sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
saver = tf.train.Saver()
saver.restore(sess, model_dir + '/saved_model.ckpt')
tags_test_flat = []
words_test = []
for s in tags_test:
    for t in s:
        tags_test_flat.append(t)
for sentence in sentences_test:
    words = ['PAD'] * 2 + sentence.lower().split(' ') + ['PAD'] * 2
    for k in range(len(words) - 4):
        words_test.append(words[k:k + WINDOW_SIZE])

pred, acc = sess.run([train_prediction, accuracy],
                     feed_dict={input_sentences: words_test,
                                output_tags: tags_test_flat})
print(acc)
write_result('../data/test.txt', pred)
sess.close()
