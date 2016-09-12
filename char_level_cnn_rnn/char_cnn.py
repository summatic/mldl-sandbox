from preprocess import quantization as qt
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
qua = qt.Quantization()


def load_raw(typ):
    col_list = []
    with open('/Users/Hanseok/PycharmProjects/ml/source/ratings_%s.txt' % typ, 'r') as f:
        for n, line in enumerate(f.readlines()[1:]):
            try:
                index, document, label = line.split('\t')
            except IndexError:
                continue
            label_list = [0, 0]
            label_list[int(label)] = 1
            col_list.append({'label': label_list, 'embedded': qua.sent_quantize(document)})
    return col_list


def batch_loader(batch_size):
    raw_data = load_raw('train')
    data_len = len(raw_data)
    if data_len % batch_size != 0:
        raise Exception("Batch size must be divisor of the length of data(%d)." % data_len)
    batch_len = int(data_len / batch_size)

    for i in range(batch_len):
        y = [data['label'] for data in raw_data[i*batch_size:(i+1)*batch_size]]
        x = [data['embedded'] for data in raw_data[i * batch_size:(i + 1) * batch_size]]
        yield x, y

# batch_size = 10000
#
# graph = tf.Graph()
# with graph.as_default():
#     train_x = tf.placeholder(tf.int8, shape=[batch_size, 336])
#     train_y = tf.placeholder(tf.int8, shape=[batch_size, 2])
#     init = tf.initialize_all_variables()
#
# iter_ = batch_loader(batch_size)
# with tf.Session(graph=graph) as sess:
#     init.run()
#     batch_x, batch_y = iter_.__next__()
#     print(len(batch_y))
#     feed_dict = {train_x: batch_x, train_y: batch_y}
#     # print(feed_dict)


class TextCNNRNN(object):

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size,
                 num_filter, filter_lengths, pool_lengths, dropout, hidden_size):
        """
        sequence_length: the length of our sentences (paddled) = 336
        num_classes: 2
        vocab_size: 111
        embedding_size: 8
        num_filter: 128
        filter_lengths: [5, 3]
        pool_lengths: [2, 2]
        dropout: 0.5
        hidden_size: 128
        """
        assert len(filter_lengths) == len(pool_lengths)
        self.input_x = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[None, num_classes], name='input_y')

        # Embedding layer
        with tf.name_scope('embedding'):
            W_embed = tf.Variable(
                tf.random_uniform(shape=[vocab_size, embedding_size], minval=-1, maxval=1),
                name='W_embed')
            embedded_chars = tf.nn.embedding_lookup(W_embed, self.input_x)
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        # Convolution + maxpool 0
        num_conv = 0
        with tf.name_scope('conv-maxpool-%d' % num_conv):
            # Conv0
            filter_shape0 = [filter_lengths[num_conv], embedding_size, 1, num_filter]
            W_conv0 = tf.Variable(tf.truncated_normal(shape=filter_shape0, stddev=0.1), name='W0')
            b_conv0 = tf.Variable(tf.constant(value=0.1, shape=[num_filter]), name='b0')
            conv0 = tf.nn.conv2d(
                input=embedded_chars_expanded,
                filter=W_conv0,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name='conv0'
            )
            # Apply nonlinearity
            h0 = tf.nn.relu(tf.nn.bias_add(conv0, b_conv0), name='relu0')
            # Max pooling
            pool0 = tf.nn.max_pool(
                value=tf.expand_dims(tf.squeeze(h0, [2]), -1),  # [?, conv, 1, num_filter] -> [?, conv, num_filter, 1]
                ksize=[1, pool_lengths[num_conv], 1, 1],
                strides=[1, pool_lengths[num_conv], 1, 1],
                padding='VALID',
                name='pool0'
            )
            num_conv += 1

        # Convolution + maxpool 1
        with tf.name_scope('conv-maxpool-%d' % num_conv):
            # Conv1
            filter_shape1 = [filter_lengths[num_conv], num_filter, 1, num_filter]
            W_conv1 = tf.Variable(tf.truncated_normal(shape=filter_shape1, stddev=0.1), name='W1')
            b_conv1 = tf.Variable(tf.constant(value=0.1, shape=[num_filter]), name='b1')
            conv1 = tf.nn.conv2d(
                input=pool0,
                filter=W_conv1,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name='conv1')
            # Apply nonlinearity
            h1 = tf.nn.relu(tf.nn.bias_add(conv1, b_conv1), name='relu1')
            # Max pooling
            pool1 = tf.nn.max_pool(
                value=tf.expand_dims(tf.squeeze(h1, [2]), -1),
                ksize=[1, pool_lengths[num_conv], 1, 1],
                strides=[1, pool_lengths[num_conv], 1, 1],
                padding='VALID',
                name='pool1')

        with tf.name_scope('drop_conv'):
            drop_conv = tf.nn.dropout(pool1, keep_prob=dropout)

        with tf.name_scope('reshape_conv_rnn'):
            _input = tf.squeeze(drop_conv, [3])
            _input = tf.transpose(_input, [1, 0, 2])
            _input = tf.reshape(_input, [-1, hidden_size])
            _input = tf.split(0, 82, _input)  # TODO: 82 대신 자동화 되서 나온 숫자.

        with tf.name_scope('Bi-LSTM'):
            istate_fw = tf.placeholder("float", [None, 2*hidden_size])
            istate_bw = tf.placeholder("float", [None, 2*hidden_size])
            lstm_fw_cell = rnn_cell.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0)
            lstm_bw_cell = rnn_cell.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0)
            output, _, _ = rnn.bidirectional_rnn(
                cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell, inputs=_input,
                initial_state_fw=istate_fw, initial_state_bw=istate_bw)

        with tf.name_scope('drop_rnn'):
            drop_rnn = tf.nn.dropout(output[-1], keep_prob=0.5)

        with tf.name_scope('softmax'):
            W_soft = tf.Variable(tf.truncated_normal(shape=[2*hidden_size, num_classes]))
            b_soft = tf.Variable(tf.constant(value=0.1, shape=[num_classes]))
            self.output = tf.matmul(drop_rnn, W_soft) + b_soft

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        cnnrnn = TextCNNRNN(sequence_length=336, num_classes=2, vocab_size=111, embedding_size=8,
                            num_filter=128, filter_lengths=[5, 3], pool_lengths=[2, 2],
                            dropout=0.5, hidden_size=128)

