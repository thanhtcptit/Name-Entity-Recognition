from __future__ import print_function
import pickle as pkl
import tensorflow as tf
from tensorflow.contrib.rnn import *
from tensorflow.python.client import device_lib
import numpy as np
from config import *
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print(get_available_gpus())
print ("Tensorflow version: " + tf.__version__)

RESTORE_DIR = 'saved_model/tmp1/'

def read_config():
    config = []
    with open(RESTORE_DIR + 'config.txt', 'r') as f:
        for line in f:
            config.append(int(line))
    return config

def get_data(emb_file, label_file):
    emb = pkl.load(open(emb_file, 'rb'))
    tag = pkl.load(open(label_file, 'rb'))

    return emb, tag

def write_result(file, prediction):
    label = ['ORG', 'PER', 'LOC', 'MISC', 'O']
    sentences = []
    sentence = []
    with open(file, 'r') as f:
        for line in f:
            if line in ['\n', '\r\n']:
                if len(sentence) == 0:
                    continue
                if len(sentence) > SENTENCE_LEN:
                    sentence = []
                    continue
                sentences.append(sentence)
                sentence = []
            else:
                tokens = line.strip().split(' ')
                assert len(tokens) == 4
                sentence.append([tokens[0], tokens[1], tokens[3], 'x'])

    for i, p in enumerate(prediction):
        last_p = -1
        for j in range(len(sentences[i])):
            if p[j] == 4:
                sentences[i][j][3] = 'O'
                last_p = -1
                continue
            if p[j] == last_p:
                sentences[i][j][3] = 'I-' + label[p[j]]
            else:
                sentences[i][j][3] = 'B-' + label[p[j]]
                last_p = p[j]


    with open(RESTORE_DIR + 'prediction.txt', 'w') as f:
        for sentence in sentences:
            for word in sentence:
                f.write(' '.join(word))
                f.write('\n')
            f.write('\n')

def padding(array, unit_size):
    global batch_size
    array_pad = [[[0] * unit_size] * batch_size] * WINDOW_SIZE + \
             array + \
             [[[0] * unit_size] * batch_size] * WINDOW_SIZE
    return array_pad

def output_fc():
    global output
    output = tf.reshape(tf.transpose(tf.stack(output), perm=[1, 0, 2]),
                        [-1, OUTPUT_SIZE])
    w_fc = tf.get_variable(name='w_fc',
                           shape=[OUTPUT_SIZE, NUM_CLASSES],
                           dtype=tf.float32)

    prediction = tf.nn.softmax(tf.matmul(output, w_fc))
    prediction = tf.reshape(prediction, [-1, SENTENCE_LEN, NUM_CLASSES])

    return prediction

def output_window_process():
    global output, batch_size
    output = padding(output, OUTPUT_SIZE)
    output = tf.transpose(tf.stack(output), perm=[1, 0, 2])

    h_hat = []
    for i in range(SENTENCE_LEN):
        output_slice = tf.slice(output, [0, i, 0],
                        [batch_size, 2 * WINDOW_SIZE + 1, OUTPUT_SIZE])

        h = tf.reshape(output_slice, [-1, (2 * WINDOW_SIZE + 1) * OUTPUT_SIZE])
        h_hat.append(h)

    h_hat = tf.reshape(tf.transpose(tf.stack(h_hat), perm=[1, 0, 2]),
                       [-1, (2 * WINDOW_SIZE + 1) * OUTPUT_SIZE])
    w_wp = tf.get_variable(name='w_wp',
                        shape=[(2 * WINDOW_SIZE + 1) * OUTPUT_SIZE, NUM_CLASSES],
                        dtype=tf.float32)
    prediction = tf.nn.softmax(tf.matmul(h_hat, w_wp))
    prediction = tf.reshape(prediction, [-1, SENTENCE_LEN, NUM_CLASSES])
    return prediction

SENTENCE_LEN, WORD_DIM,  NUM_RNN_LAYERS, HIDDEN_SIZE, \
    KEEP_PR, WINDOW_SIZE = read_config()
OUTPUT_SIZE = 2 * HIDDEN_SIZE


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

x_test, y_test = get_data('data/x_test_' + str(SENTENCE_LEN) + '.pkl',
                          'data/y_test_' + str(SENTENCE_LEN) + '.pkl')
batch_size = len(x_test)
input_data = tf.placeholder(tf.float32, [None, SENTENCE_LEN, WORD_DIM])
output_data = tf.placeholder(tf.float32, [None, SENTENCE_LEN, NUM_CLASSES])

# Network architecture
'''
fw_cell = tf.contrib.rnn.MultiRNNCell([DropoutWrapper(
                GRUCell(hidden_size), output_keep_prob=keep_pr)
                for _ in range(num_layers)], state_is_tuple=True)
bw_cell = tf.contrib.rnn.MultiRNNCell([DropoutWrapper(
                GRUCell(hidden_size), output_keep_prob=keep_pr)
                for _ in range(num_layers)], state_is_tuple=True)
'''
# LSTM
fw_cell = tf.contrib.rnn.MultiRNNCell([DropoutWrapper(LSTMCell(
                HIDDEN_SIZE, state_is_tuple=True), output_keep_prob=KEEP_PR)
                for _ in range(NUM_RNN_LAYERS)], state_is_tuple=True)
bw_cell = tf.contrib.rnn.MultiRNNCell([DropoutWrapper(LSTMCell(
                HIDDEN_SIZE, state_is_tuple=True), output_keep_prob=KEEP_PR)
                for _ in range(NUM_RNN_LAYERS)], state_is_tuple=True)

# count actual word in sentence (not padding)
words_used_in_sent = tf.sign(tf.reduce_max(tf.abs(input_data),
                                           reduction_indices=2))
length = tf.cast(tf.reduce_sum(words_used_in_sent,
                               reduction_indices=1), tf.int32)
inputs = tf.unstack(tf.transpose(input_data, perm=[1, 0, 2]))
output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell,
                inputs, dtype=tf.float32, sequence_length=length)

prediction = output_window_process()

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#saver.restore(sess, 'saved_model/' + str(sentence_len) +'/saved_model.ckpt')
saver.restore(sess, RESTORE_DIR + 'saved_model.ckpt')
p, p_length = sess.run([prediction, length],
                            {input_data: x_test,
                            output_data: y_test})
p = np.argmax(p, 2)
write_result('data/test.txt', p)
print ('Done')