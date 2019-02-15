import tensorflow as tf
from tensorflow.python.client import device_lib
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

def get_data(sentence_file, tag_file):
    with open(sentence_file, 'r') as f:
        sentences = f.read().split('\n')
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
    return vocab, reversed_vocab

def sentence_to_id(sentence, vocab, max_len):
    tokens = sentence.split(' ')
    sentence_len = len(tokens)
    pad_length = max_len - sentence_len
    if WINDOW_SIZE > 1:
        return [0] * 2 + [vocab[t] if t in vocab else vocab['<UNK>']
                for t in tokens] + [0] * (pad_length + 2), sentence_len
    else:
        return [vocab[t] if t in vocab else vocab['<UNK>']
                for t in tokens] + [0] * pad_length, sentence_len

def id_to_sentence(idx, reversed_vocab):
    return ' '.join([reversed_vocab[i] for i in idx])

def transform(sentence, vocab):
    x_transform = []
    for s in sentence:
        sent_idx, length = sentence_to_id(s, vocab, MAX_SENTENCE_LEN)
        x_transform.append(sent_idx)
    return x_transform[:-1]

def save_params(model_dir):
    with open(model_dir + '/params.txt', 'w') as f:
        f.write(str(EMBEDDING_SIZE) + '\n')
        f.write(str(WINDOW_SIZE) + '\n')
        f.write(' '.join(np.array(FILTER_SIZE, dtype=np.string_)) + '\n')
        f.write(str(NUM_FILTER) + '\n')

model_dir = 'model_1'
model_max_dir = 'model_2'

sentences_train, tags_train = get_data('../data/sentences_train.txt',
                           '../data/y_train_124.pkl')
sentences_val, tags_val = get_data('../data/sentences_val.txt',
                           '../data/y_val_124.pkl')
vocab, reversed_vocab = get_vocab('../data/vocab.txt')

# Data information
MAX_SENTENCE_LEN = 124
NUM_CLASSES = 9
TRAIN_SIZE = len(sentences_train)
VOCAB_SIZE = len(vocab)

# Hyper params
EMBEDDING_SIZE = 100
WINDOW_SIZE = 5
FILTER_SIZE = [3, 4, 5]
NUM_FILTER = 128
TOTAL_FILTERS = NUM_FILTER * len(FILTER_SIZE)

# Trainning params
BATCH_SIZE = 10
NUM_EPOCH = 200
LEARNING_RATE = 0.001
KEEP_PROB = 0.75
L2_REG = 0.01
NUM_BATCH = int((TRAIN_SIZE - 1) / BATCH_SIZE) + 1

# Prepare Data
x_train = transform(sentences_train, vocab)
x_val = transform(sentences_val, vocab)
print ('Training size: ', TRAIN_SIZE)
x_shuffled, y_shuffled = shuffle(x_train, tags_train)

# Graph
input_sentences = tf.placeholder(tf.int32,
                    [None, WINDOW_SIZE],
                    'input_sentences')
output_tags = tf.placeholder(tf.int32,
                    [None, NUM_CLASSES],
                    'output_tags')
keep_pr = tf.placeholder(tf.float32, name='keep_pr')

# Trainable variable
embedding_matrix = tf.Variable(
                tf.random_uniform([VOCAB_SIZE, EMBEDDING_SIZE], -1.0, 1.0),
                name="embedding_matrix")

input_vector = tf.nn.embedding_lookup(
    embedding_matrix, input_sentences, 'input_vector')
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
    conv_output = tf.nn.dropout(conv_output, keep_pr)
    fc = tf.matmul(conv_output, fc_weight)
    regularizer = tf.nn.l2_loss(fc_weight)
    return fc, regularizer

# Predict
prediction, reg = apply_CNN(input_vector)
train_prediction = tf.argmax(prediction, 1, name="predictions")
correct_predictions = tf.equal(train_prediction,
                        tf.argmax(output_tags, 1))
accuracy = tf.reduce_mean(
    tf.cast(correct_predictions, "float"), name="accuracy")
loss = tf.nn.softmax_cross_entropy_with_logits(
    labels=output_tags, logits=prediction)
loss = tf.reduce_mean(loss + L2_REG * reg)
#tf.summary.scalar("loss", loss)

# GD
global_step = tf.Variable(0, name="global_step", trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_vars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, train_vars), 10)
train_op = optimizer.apply_gradients(zip(grads, train_vars), global_step)

#summary_op = tf.summary.merge_all()
# Trainning
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
save_params(model_dir)
save_params(model_max_dir)
#writer = tf.summary.FileWriter('../log/', graph=tf.get_default_graph())
words_val = []
tags_val = np.reshape(tags_val, [-1, NUM_CLASSES])
for sentence in x_val:
    for k in range(len(sentence)):
        words_val.append(sentence[k:k + WINDOW_SIZE])
best_acc = 0
for epoch in range(1, NUM_EPOCH + 1):
    epoch_loss = 0
    for j in range(NUM_BATCH):
        sentences_batch = x_shuffled[j * BATCH_SIZE:
            j * BATCH_SIZE + BATCH_SIZE]
        tags_batch = y_shuffled[j * BATCH_SIZE:
            j * BATCH_SIZE + BATCH_SIZE]
        tags_batch = p.reshape(tags_batch, [-1, NUM_CLASSES])
        words_batch = []
        for sentence in sentences_batch:
            for k in range(len(sentence)):
                words_batch.append(sentence[k:k + WINDOW_SIZE])
        words_batch, tags_batch = shuffle(words_batch, tags_batch)
        _, current_step, batch_loss, batch_acc = sess.run(
                        [train_op, global_step, loss, accuracy],
                        feed_dict={input_sentences:words_batch,
                                   output_tags:tags_batch,
                                   keep_pr:KEEP_PROB})
        #writer.add_summary(summary, current_step)
        #print ('Step: {} - Loss: {} - Accuracy: {}'.format(
        #    current_step, batch_loss, batch_acc))
        epoch_loss += batch_loss

    print ('EPOCH: {} - Loss: {}'.format(epoch, epoch_loss))
    if epoch % 5 == 0:
        val_accuracy = sess.run([accuracy],
                        feed_dict={input_sentences:words_val,
                                   output_tags:tags_val,
                                   keep_pr:1.0})
        saver.save(sess, model_dir + '/saved_model.ckpt')
        print ("Validation accuracy: ", val_accuracy)
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            saver.save(sess, model_max_dir + '/saved_model.ckpt')
print ('Best acc: ', best_acc)
sess.close()