import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.contrib.rnn import *
import pickle as pkl
import numpy as np
from tensorflow.contrib import learn
from sklearn.utils import shuffle
import argparse

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos
            if x.device_type == 'GPU']

print (get_available_gpus())
print ('Tensorflow version: ', tf.__version__)
print ('Numpy version: ', np.__version__)

config = tf.ConfigProto(device_count = {'GPU': 1})
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def get_data(words_file, tag_file):
    words = pkl.load(open(words_file, 'rb'))
    tags = pkl.load(open(tag_file, 'rb'))
    return words, tags

def get_char_vocab(file):
    vocab = {}
    reversed_vocab = {}
    with open(file, 'r') as f:
        items = f.read().split('\n')
        for i, w in enumerate(items):
            vocab[w] = i
            reversed_vocab[i] = w
    return vocab, reversed_vocab

def word_to_id(word, vocab):
    word_len = len(word)
    padding = MAX_WORD_LEN - word_len
    return [vocab[t] if t in vocab else vocab['<UNK>']
            for t in word] + [0] * padding, word_len

def id_to_word(idx, reversed_vocab):
    return ' '.join([reversed_vocab[i] for i in idx])

def transform(words, vocab):
    x_transform = []
    for w in words:
        word_idx, length = word_to_id(w, vocab)
        x_transform.append(word_idx)
    return x_transform

def load_params(model_dir):
    params = []
    with open(model_dir + '/params.txt', 'r') as f:
        for i, line in enumerate(f):
            if i == 2 or i == 3:
                params.append(np.array(line.split(' '), dtype=np.int32))
            else:
                params.append(int(line))
    return params

def write_result(file, prediction):
    label = ['B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER',
             'B-MISC', 'I-MISC', 'O']
    words = []
    true_label = []
    with open(file, 'r') as f:
        for line in f:
            if line in ['\n', '\r\n']:
                words.append(line)
                continue
            tokens = line.strip().split(' ')
            assert len(tokens) == 4
            words.append([tokens[0], tokens[1], tokens[3], 'x'])
            true_label.append(label.index(tokens[3]))
    assert len(true_label) == len(prediction)
    confusion_matrix(prediction, true_label)
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

def confusion_matrix(pred, true_label):
    cm = [[0] * len(pred)] * len(pred)
    for i in range(len(pred)):
        cm[pred[i]][true_label[i]] += 1
    print (cm)


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str)

model_dir = parser.parse_args().model_dir
words_test, tags_test = get_data('../data/x_test_v1.pkl',
                           '../data/y_test_v1.pkl')
vocab, reversed_vocab = get_char_vocab('../data/char-vocab.txt')

# Data information
MAX_WORD_LEN = 27
NUM_CLASSES = 9
TEST_SIZE = len(words_test)
VOCAB_SIZE = len(vocab)
print ('Test size: ', TEST_SIZE)

# Hyper params
EMBEDDING_SIZE = 15
FILTER_SIZE = [1, 2, 3, 4, 5, 6]
NUM_FILTER = [25, 50, 75, 100, 125, 150]
TOTAL_FILTERS = sum(NUM_FILTER)
LSTM_HIDDEN_UNIT = 300
LSTM_NUM_LAYER = 2
WORD_PER_BATCH = 35
BATCH_SIZE = ((TEST_SIZE - 1) / WORD_PER_BATCH) + 1
PADDING_WORD = (BATCH_SIZE * WORD_PER_BATCH) - TEST_SIZE

x_test = transform(words_test, vocab)

input_sentences = tf.placeholder(tf.int32,
                    [None, MAX_WORD_LEN],
                    'input_sentences')
output_tags = tf.placeholder(tf.int32,
                    [None, NUM_CLASSES],
                    'output_tags')

embedding_matrix = tf.Variable(
                tf.random_uniform([VOCAB_SIZE, EMBEDDING_SIZE], -1.0, 1.0),
                name="embedding_matrix")

input_vector = tf.nn.embedding_lookup(
    embedding_matrix, input_sentences, 'input_vector')
input_vector = tf.expand_dims(input_vector, -1)

def conv2d(x, W, b, strides=1):
    x1 = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1],
                     padding='VALID', name='conv2d')
    x1 = tf.nn.bias_add(x1, b)
    return tf.nn.relu(x1, name='relu')

def max_pool2d(x, k, stride=1):
    return tf.nn.max_pool(x, ksize=[1, k[0], k[1], 1],
                          strides=[1, stride, stride, 1],
                          padding='VALID', name='max_pool2d')

def cnn_layer(x):
    filter_weights = {
        'fw' + str(i + 1): tf.Variable(tf.random_normal(
            [FILTER_SIZE[i], EMBEDDING_SIZE, 1, NUM_FILTER[i]], stddev=0.1),
            name='fw' + str(i + 1)) for i in range(len(FILTER_SIZE))
    }

    filter_bias = {
        'fb' + str(i + 1): tf.Variable(
            tf.constant(0.1, shape=[NUM_FILTER[i]]), name='fb' + str(i + 1))
        for i in range(len(FILTER_SIZE))
    }
    conv_output = []
    for i in range(len(FILTER_SIZE)):
        conv = conv2d(x, filter_weights['fw' + str(i + 1)],
                    filter_bias['fb' + str(i + 1)])
        conv_output.append(max_pool2d(conv,
                    [MAX_WORD_LEN - FILTER_SIZE[i] + 1, 1]))
    conv_output = tf.reshape(tf.concat(conv_output, 3),
                             [-1, TOTAL_FILTERS])
    #conv_output = tf.nn.dropout(conv_output, keep_pr)
    return conv_output

def highway_layer(x):
    t_weight = tf.get_variable(shape=[TOTAL_FILTERS, TOTAL_FILTERS],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        name='t_weights')
    t_bias = tf.get_variable(initializer=tf.constant(0.1, shape=[TOTAL_FILTERS]),
                             name='t_bias')
    transform_gate = tf.nn.sigmoid(tf.matmul(x, t_weight) + t_bias)
    h_weight = tf.get_variable(shape=[TOTAL_FILTERS, TOTAL_FILTERS],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        name='h_weights')
    h_bias = tf.get_variable(initializer=tf.constant(0.1, shape=[TOTAL_FILTERS]),
                             name='h_bias')
    highway_output = transform_gate * tf.tanh(tf.matmul(x, h_weight) + h_bias) \
                        + (1 - transform_gate) * x
    return highway_output

def lstm_layer(x):
    fw_cell = tf.contrib.rnn.MultiRNNCell([DropoutWrapper(
        LSTMCell(LSTM_HIDDEN_UNIT)) 
        for _ in range(LSTM_NUM_LAYER)], state_is_tuple=True)
    #bw_cell = tf.contrib.rnn.MultiRNNCell([DropoutWrapper(LSTMCell(
    #    LSTM_HIDDEN_UNIT), output_keep_prob=keep_pr)
    #    for _ in range(LSTM_NUM_LAYER)], state_is_tuple=True)
    x = tf.reshape(x, [BATCH_SIZE, WORD_PER_BATCH, TOTAL_FILTERS])
    x = tf.unstack(tf.transpose(x, [1, 0, 2]))
    output, _ = tf.contrib.rnn.static_rnn(fw_cell, x, dtype=tf.float32)
    output = tf.reshape(tf.unstack(tf.transpose(tf.stack(output), perm=[1, 0, 2])),
                            [-1, LSTM_HIDDEN_UNIT])
    return output

def fc_layer(x):
    fc_weight = tf.get_variable(shape=[LSTM_HIDDEN_UNIT, NUM_CLASSES],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                name='fc_weight')
    output = tf.matmul(x, fc_weight)
    return output

input_vector = cnn_layer(input_vector)
input_vector = highway_layer(input_vector)
input_vector = lstm_layer(input_vector)
prediction = fc_layer(input_vector)
prediction = tf.argmax(prediction, 1, name="predictions")
correct_predictions = tf.equal(prediction,
                        tf.argmax(output_tags, 1))
accuracy = tf.reduce_mean(
    tf.cast(correct_predictions, "float"), name="accuracy")

# Trainning
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, model_dir + '/saved_model.ckpt')
for i in range(PADDING_WORD):
    x_test.append(MAX_WORD_LEN * [0])
    tags_test.append(NUM_CLASSES * [0])
pred, acc = sess.run([prediction, accuracy],
            feed_dict={input_sentences:x_test,
                       output_tags:tags_test})
print (acc)
write_result('../data/test.txt', pred[:TEST_SIZE])
sess.close()
