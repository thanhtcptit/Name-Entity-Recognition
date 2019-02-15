import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.contrib.rnn import *
import pickle as pkl
import numpy as np
from tensorflow.contrib import learn
from sklearn.utils import shuffle

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

def save_params(model_dir):
    with open(model_dir + '/params.txt', 'w') as f:
        f.write(str(EMBEDDING_SIZE) + '\n')
        f.write(' '.join(np.array(FILTER_SIZE, dtype=np.string_)) + '\n')
        f.write(' '.join(np.array(FILTER_SIZE, dtype=np.string_)))

model_dir = 'model_1'
model_max_dir = 'model_2'
words_train, tags_train = get_data('../data/sample/x_train.pkl',
                           '../data/sample/y_train.pkl')
vocab, reversed_vocab = get_char_vocab('../data/sample/char-vocab.txt')

# Data information
MAX_WORD_LEN = 27
NUM_CLASSES = 9
TRAIN_SIZE = len(words_train)
VOCAB_SIZE = len(vocab)
print ('Train size: ', TRAIN_SIZE)

# Hyper params
EMBEDDING_SIZE = 15
FILTER_SIZE = [1, 2, 3, 4, 5, 6]
NUM_FILTER = [25, 50, 75, 100, 125, 150]
TOTAL_FILTERS = sum(NUM_FILTER)
LSTM_HIDDEN_UNIT = 300
LSTM_NUM_LAYER = 2

# Trainning params
BATCH_SIZE = 2
WORD_PER_BATCH = 5
NUM_WORD_INPUT = BATCH_SIZE * WORD_PER_BATCH
NUM_EPOCH = 200
LEARNING_RATE = 0.001
KEEP_PROB = 0.75
L2_REG = 0.01
NUM_BATCH = int((TRAIN_SIZE - 1) / (NUM_WORD_INPUT)) + 1

x_train = transform(words_train, vocab)
x_shuffled, y_shuffled = shuffle(x_train, tags_train)

input_sentences = tf.placeholder(tf.int32,
                    [None, MAX_WORD_LEN],
                    'input_sentences')
output_tags = tf.placeholder(tf.int32,
                    [None, NUM_CLASSES],
                    'output_tags')
keep_pr = tf.placeholder(tf.float32, name='keep_pr')

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
    highway_output = transform_gate * tf.nn.relu(tf.matmul(x, h_weight) + h_bias) \
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
    output = tf.nn.dropout(output, keep_pr)
    return output

def fc_layer(x):
    fc_weight = tf.get_variable(shape=[LSTM_HIDDEN_UNIT, NUM_CLASSES],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                name='fc_weight')
    output = tf.matmul(x, fc_weight)
    regularizer = tf.nn.l2_loss(fc_weight)
    return output, regularizer

input_vector = cnn_layer(input_vector)
input_vector = highway_layer(input_vector)
input_vector = lstm_layer(input_vector)
prediction, reg = fc_layer(input_vector)
train_prediction = tf.argmax(prediction, 1, name="predictions")
correct_predictions = tf.equal(train_prediction,
                        tf.argmax(output_tags, 1))
accuracy = tf.reduce_mean(
    tf.cast(correct_predictions, "float"), name="accuracy")
loss = tf.nn.softmax_cross_entropy_with_logits(
    labels=output_tags, logits=prediction)
loss = tf.reduce_mean(loss + L2_REG * reg)

# GD
global_step = tf.Variable(0, name="global_step", trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_vars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, train_vars), 10)
train_op = optimizer.apply_gradients(zip(grads, train_vars), global_step)

# Trainning
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
best_acc = 0
for epoch in range(1, NUM_EPOCH + 1):
    epoch_loss = 0
    for j in range(NUM_BATCH):
        start_idx = j * (NUM_WORD_INPUT)
        end_idx = j * (NUM_WORD_INPUT) + NUM_WORD_INPUT
        if end_idx >= TRAIN_SIZE:
            end_idx = TRAIN_SIZE
            start_idx = end_idx - NUM_WORD_INPUT
        words_batch = x_shuffled[start_idx:end_idx]
        tags_batch = y_shuffled[start_idx:end_idx]
        words_batch, tags_batch = shuffle(words_batch, tags_batch)
        _, current_step, batch_loss, batch_pred = sess.run(
                        [train_op, global_step, loss, train_prediction],
                        feed_dict={input_sentences:words_batch,
                                   output_tags:tags_batch,
                                   keep_pr:KEEP_PROB})
        print (batch_pred)
        exit()
        print ('Step: {} - Loss: {} - Accuracy: {}'.format(
            current_step, batch_loss, batch_acc))
        epoch_loss += batch_loss

    print ('EPOCH: {} - Loss: {}'.format(epoch, epoch_loss))
sess.close()
