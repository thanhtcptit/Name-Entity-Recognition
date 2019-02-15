from __future__ import print_function
import pickle as pkl
import tensorflow as tf
from tensorflow.contrib.rnn import *
from tensorflow.python.client import device_lib
import numpy as np

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
print (get_available_gpus())

def get_data(emb_file, label_file):
    emb = pkl.load(open(emb_file, 'rb'))
    tag = pkl.load(open(label_file, 'rb'))

    return emb, tag

# Data infomation
sentence_len = 10
word_dim = 61

# Hyper params
n_epoch = 200
batch_size = 2
lr = 0.001
num_class = 5
num_layers = 2
hidden_size = 128
keep_pr = 0.8
context_window = 2
context_vector_len = 128

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session()

train_data, y_train = get_data('sample/data/x_train.pkl', 'sample/data/y_train.pkl')
x_train = []
for sentence in train_data:
    sentence = [[0] * word_dim] * context_window + \
               sentence + [[0] * word_dim] * context_window
    context_sentence = []
    for i in range(2, len(sentence) - 2):
        context_word = []
        for pi in range(-context_window, context_window + 1):
            context_word.extend(sentence[i + pi])
        context_sentence.append(context_word)
    x_train.append(context_sentence)

x_train = tf.constant(x_train, dtype=tf.float32)
y_train = tf.constant(y_train, dtype=tf.float32)
num_train = x_train.shape[0]
num_batch = int(num_train / batch_size)
batch_index = tf.placeholder(tf.int32, shape=[batch_size])
# input_data: batch x sentence_len x word_dim
input_data = tf.gather(x_train, indices=batch_index)
# output_data: batch x sentence_len x num_class
output_data = tf.gather(y_train, indices=batch_index)

# Network architecture
fw_cell = tf.contrib.rnn.MultiRNNCell([DropoutWrapper(GRUCell(
                hidden_size), output_keep_prob=keep_pr)
                    for _ in range(num_layers)], state_is_tuple=True)
bw_cell = tf.contrib.rnn.MultiRNNCell([DropoutWrapper(GRUCell(
                    hidden_size), output_keep_prob=keep_pr)
                    for _ in range(num_layers)], state_is_tuple=True)

# count actual word in sentence (not padding)
words_used_in_sent = tf.sign(tf.reduce_max(tf.abs(input_data),
                    reduction_indices=2))
length = tf.cast(tf.reduce_sum(words_used_in_sent,
                    reduction_indices=1), tf.int32)
# inputs : list(sentence_len) (batch x word_dim)
inputs = tf.unstack(tf.transpose(input_data, perm=[1, 0, 2]))
# output : list(sentence_len) (batch x 2 * hidden_size)
output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell,
                inputs, dtype=tf.float32, sequence_length=length)
# (batch * sentence_len) x (2 * hidden_size)
output = tf.reshape(tf.unstack(tf.transpose(tf.stack(output), perm=[1, 0, 2])),
                    [-1, 2 * hidden_size])
# Attention
'''
u_w = tf.get_variable('u_w', shape=[context_vector_len, tmp1])
w_w = tf.get_variable('w_w',
            shape=[2 * hidden_size, context_vector_len], dtype=tf.float32)
b_w = tf.get_variable('b_w', shape=[context_vector_len], dtype=tf.float32)
u_t = tf.tanh(tf.matmul(output, w_w) + b_w)
# batch x sentence_len x context_vector_len
u_t = tf.reshape(u_t, [-tmp1, sentence_len, context_vector_len])

# list(batch_size) (sentence_len x tmp1)
score = [tf.nn.softmax(tf.matmul(u_t[batch], u_w), dim=0)
         for batch in range(u_t.shape[0])]
score = tf.reshape(tf.stack(score), [batch_size * sentence_len, tmp1])
h_hat = score * output
h_hat = tf.reshape(h_hat, [-tmp1, sentence_len, 2 * hidden_size])

final_output = [[0] * sentence_len] * batch_size
for i in range(batch_size):
    for j in range(sentence_len):
        if j == 0:
            final_output[i][j] = h_hat[i][j] + h_hat[i][j + tmp1] + h_hat[i][j + 2]
        elif j == tmp1:
            final_output[i][j] = h_hat[i][j - tmp1] + h_hat[i][j] + \
                h_hat[i][j + tmp1] + h_hat[i][j + 2]
        elif j == sentence_len - 2:
            final_output[i][j] = h_hat[i][j - 2] + h_hat[i][j - tmp1] + \
                h_hat[i][j] + h_hat[i][j + tmp1]
        elif j == sentence_len - tmp1:
            final_output[i][j] = h_hat[i][j - 2] + h_hat[i][j - tmp1] + \
                h_hat[i][j]
        else:
            final_output[i][j] = h_hat[i][j - 2] + h_hat[i][j - tmp1] + \
                h_hat[i][j] + h_hat[i][j + tmp1] + h_hat[i][j + 2]

final_output = tf.reshape(tf.stack(final_output), [-tmp1, 2 * hidden_size])
'''


w_fc = tf.get_variable(name='w_fc',
            shape=[2 * hidden_size, num_class], dtype=tf.float32)

prediction = tf.matmul(output, w_fc)
prediction = tf.reshape(prediction, [-1, sentence_len, num_class])

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=output_data, logits=prediction))

optimizer = tf.train.AdamOptimizer(learning_rate=lr)

train_vars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, train_vars), 10)
train_op = optimizer.apply_gradients(zip(grads, train_vars))

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
n_idx = list(range(num_train))
mini_batch = np.array_split(n_idx, num_batch)[:-1]
for epoch in range(n_epoch + 1):
    epoch_loss = 0
    for batch in mini_batch:
        batch_loss, _ = sess.run([loss, train_op],
                        feed_dict={batch_index: batch})
        epoch_loss += batch_loss

    #if epoch % 200 == 0:
    #    save_path = saver.save(sess, 'saved_model/saved_model.ckpt')
    #    print('Model saved in file: ', save_path)
    print('Epoch %d - Loss: %f ' % (epoch, epoch_loss))
