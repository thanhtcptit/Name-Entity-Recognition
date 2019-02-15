import tensorflow as tf
from tensorflow.python.client import device_lib
import pickle as pkl
import numpy as np

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
print (get_available_gpus())
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
        return [0] * 2 + [vocab[t] for t in tokens] + [0] * (pad_length + 2), \
               sentence_len
    else:
        return [vocab[t] for t in tokens] + [0] * pad_length, sentence_len

def id_to_sentence(idx, reversed_vocab):
    return ' '.join([reversed_vocab[i] for i in idx])

def load_embedding_model():
    model = {}
    with open("../data/glove.6B.100d.txt", 'r') as f:
        for line in f:
            tokens = line.replace('\n', '').split(' ')
            model[tokens[0]] = np.array(tokens[1:], dtype=np.float32).tolist()
    return np.array(model)


# Hyper params
EMBEDDING_SIZE = 100
WINDOW_SIZE = 5
NUM_FILTER = 50
FILTER_SIZE = 1
LEARNING_RATE = 0.001

# Prepare Data
sentences, tags = get_data('../data/train_sentences.txt',
                           '../data/y_train_30.pkl')
vocab, reversed_vocab, vocab_size = get_vocab('../data/vocab.txt')
train_size = len(sentences)
print ('Training size: ', train_size)
batch_size = 64
num_batch = int(train_size / batch_size)
max_sentence_len = 30
total_word = num_batch * batch_size * max_sentence_len
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
    'fw1' : tf.Variable(tf.random_normal([5, 5, 1, NUM_FILTER])),
    'fc1' : tf.Variable(tf.random_normal([(WINDOW_SIZE - FILTER_SIZE + 1)
                                          * NUM_FILTER, num_classes])),
    'fc2' : tf.Variable(tf.random_normal([96 * 50, num_classes]))
}
filter_bias = {
    'fb1': tf.Variable(tf.random_normal([NUM_FILTER])),
    'fc1': tf.Variable(tf.random_normal([num_classes]))
}

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1],
                     padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def apply_CNN(x, drop_out=0.5):
    x = tf.reshape(x, shape=[-1, WINDOW_SIZE, EMBEDDING_SIZE, 1])
    conv1 = conv2d(x, filter_weights['fw1'], filter_bias['fb1'])
    fc = tf.reshape(conv1, [-1, filter_weights['fc2'].get_shape().as_list()[0]])
    #fc = tf.nn.dropout(fc, drop_out)
    fc = tf.matmul(fc, filter_weights['fc2']) + filter_bias['fc1']
    #fc = tf.matmul(fc, filter_weights['fc2'])
    return fc

# Predict
prediction = []
for i in range(max_sentence_len):
    word = tf.slice(input_vector, [0, i, 0],
                    [-1, WINDOW_SIZE, EMBEDDING_SIZE])
    prediction.append(apply_CNN(word))

prediction = tf.transpose(tf.stack(prediction), perm=[1, 0, 2])
train_prediction = tf.nn.softmax(prediction)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=output_tags, logits=prediction))
# GD
optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
train_vars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, train_vars), 10)
train_op = optimizer.apply_gradients(zip(grads, train_vars))

# Trainning
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
best_acc = 0
for epoch in range(1, num_epoch + 1):
    epoch_loss = 0
    tp = 0
    for i in range(num_batch):
        sentence_batch = sentences[i * batch_size: i * batch_size + batch_size]
        output_batch = tags[i * batch_size: i * batch_size + batch_size]
        input_batch = []
        for sentence in sentence_batch:
            idx, length = sentence_to_id(sentence, vocab, max_sentence_len)
            input_batch.append(idx)

        batch_loss, _, batch_prediction = sess.run(
                        [loss, train_op, train_prediction],
                        feed_dict={input_sentences:input_batch,
                                   output_tags:output_batch})
        epoch_loss += batch_loss
        tp += np.sum(np.argmax(batch_prediction, 2)
                     == np.argmax(output_batch, 2))
    train_accuracy = (float(tp) / total_word) * 100
    print ('Epoch: {} - Loss: {} - Accuracy: {}'.
           format(epoch, epoch_loss, train_accuracy))
    if epoch % 10 == 0:
        save_path = saver.save(sess, 'saved_model/tmp1/saved_model.ckpt')
        print('Model saved in file: ', save_path)
        if train_accuracy > best_acc:
            saver.save(sess, 'saved_model/tmp2/saved_model.ckpt')
            best_acc = train_accuracy
print ('Best accuracy: ', best_acc)
sess.close()

