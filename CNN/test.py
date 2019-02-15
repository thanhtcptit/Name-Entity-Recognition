from __future__ import print_function
import tensorflow as tf
from tensorflow.python.client import device_lib
import pickle as pkl
import numpy as np

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
print ('GPU: ', get_available_gpus())
print ('Tensorflow version: ', tf.__version__)
print ('Numpy version: ', np.__version__)

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
            vocab[w] = i
            reversed_vocab[i] = w
    return vocab, reversed_vocab, len(vocab)

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

def write_result(file, prediction):
    label = ['B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 
	'B-MISC', 'I-MISC', 'O']
    sentences = []
    sentence = []
    with open(file, 'r') as f:
        for line in f:
            if line in ['\n', '\r\n']:
                if len(sentence) == 0:
                    continue
                if len(sentence) > max_sentence_len:
                    sentence = []
                    continue
                sentences.append(sentence)
                sentence = []
            else:
                tokens = line.strip().split(' ')
                assert len(tokens) == 4
                sentence.append([tokens[0], tokens[1], tokens[3], 'x'])

    for i, p in enumerate(prediction):
        for j in range(len(sentences[i])):
            sentences[i][j][3] = label[p[j]]

    with open('saved_model/tmp2/prediction.txt', 'w') as f:
        for sentence in sentences:
            for word in sentence:
                f.write(' '.join(word))
                f.write('\n')
            f.write('\n')

# Hyper params
EMBEDDING_SIZE = 100
WINDOW_SIZE = 5
NUM_FILTER = 50
FILTER_SIZE = 5

# Prepare Data
sentences, tags = get_data('data/sentences_test.txt',
                           'data/y_test_30.pkl')
vocab, reversed_vocab, vocab_size = get_vocab('data/vocab.txt')
train_size = len(sentences)
print ('Training size: ', train_size)
batch_size = train_size
num_batch = int(train_size / batch_size)
max_sentence_len = 30
num_classes = 9
num_epoch = 200

input_sentences = tf.placeholder(tf.int32,
                    [None, max_sentence_len + WINDOW_SIZE - 1],
                    'input_sentences')
output_tags = tf.placeholder(tf.int32,
                    [None, max_sentence_len, num_classes],
                    'output_tags')
# Trainable variable
embedding_matrix = tf.get_variable('embedding_matrix',
    [vocab_size, EMBEDDING_SIZE])

input_vector = tf.nn.embedding_lookup(embedding_matrix, input_sentences,
                          'input_vector')

filter_weights = {
    'fw1' : tf.Variable(tf.random_normal([FILTER_SIZE, 100, 1, NUM_FILTER])),
    'out' : tf.Variable(tf.random_normal([(WINDOW_SIZE - FILTER_SIZE + 1) *
                                          NUM_FILTER, num_classes]))
}
filter_bias = {
    'fb1': tf.Variable(tf.random_normal([NUM_FILTER])),
}

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1],
             padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def apply_CNN(x, drop_out=0.5):
    x = tf.reshape(x, shape=[-1, WINDOW_SIZE, EMBEDDING_SIZE, 1])
    conv1 = conv2d(x, filter_weights['fw1'], filter_bias['fb1'])
    fc = tf.reshape(conv1, [-1, filter_weights['out'].get_shape().as_list()[0]])
    #fc = tf.nn.dropout(fc, drop_out)
    out = tf.nn.softmax(tf.matmul(fc, filter_weights['out']))
    return out

# Predict
prediction = []
for i in range(max_sentence_len):
    word = tf.slice(input_vector, [0, i, 0],
            [batch_size, WINDOW_SIZE, EMBEDDING_SIZE])
    prediction.append(apply_CNN(word))
prediction = tf.transpose(tf.stack(prediction), perm=[1, 0, 2])

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, 'saved_model/tmp2/saved_model.ckpt')
input_batch = []
for sentence in sentences:
    idx, length = sentence_to_id(sentence, vocab, max_sentence_len)
    input_batch.append(idx)

p = sess.run([prediction], feed_dict={input_sentences:input_batch,
                                output_tags:tags})
p = np.argmax(p, 2)
print (p.shape)
exit()
write_result('data/preprocessed/test.txt', p)
sess.close()

