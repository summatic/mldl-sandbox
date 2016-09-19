import tensorflow as tf
import math
from tensorflow.python.ops import rnn, rnn_cell


class TextCNNRNN(object):

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size,
                 num_filter, filter_lengths, pool_lengths, dropout, hidden_size, l2_reg_lambda):
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
        l2_reg_lambda: 5 * 10 ^ -4
        """
        assert len(filter_lengths) == len(pool_lengths)

        # Placeholders for input, output and initial state of bi-lstm
        input_x = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_x')
        input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        istate_fw = tf.placeholder("float", [None, 2 * hidden_size])
        istate_bw = tf.placeholder("float", [None, 2 * hidden_size])

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.name_scope('embedding'):
            W_embed = tf.Variable(
                tf.random_uniform(shape=[vocab_size, embedding_size], minval=-1, maxval=1),
                name='W_embed')
            embedded_chars = tf.nn.embedding_lookup(W_embed, input_x)
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

        # Add dropout 0
        with tf.name_scope('drop_conv'):
            drop_conv = tf.nn.dropout(pool1, keep_prob=dropout)

        # Reshape tensor
        with tf.name_scope('reshape_conv_rnn'):
            _input = tf.squeeze(drop_conv, [3])
            _input = tf.transpose(_input, [1, 0, 2])
            _input = tf.reshape(_input, [-1, hidden_size])
            _input = tf.split(0, 82, _input)  # TODO: 82 대신 자동화 되서 나온 숫자.

        # Bidirectional
        with tf.name_scope('Bi-LSTM'):
            lstm_fw_cell = rnn_cell.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0)
            lstm_bw_cell = rnn_cell.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0)
            output, _, _ = rnn.bidirectional_rnn(
                cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell, inputs=_input,
                initial_state_fw=istate_fw, initial_state_bw=istate_bw)

        # Add dropout1
        with tf.name_scope('drop_rnn'):
            drop_rnn = tf.nn.dropout(output[-1], keep_prob=0.5)

        # Final (unnormalized) scores and predictions
        with tf.name_scope('softmax'):
            W_soft = tf.get_variable(
                'W', shape=[2*hidden_size, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            # W_soft = tf.Variable(tf.truncated_normal(shape=[2*hidden_size, num_classes]))
            b_soft = tf.Variable(tf.constant(value=0.1, shape=[num_classes]))
            l2_loss += tf.nn.l2_loss(W_soft)
            l2_loss += tf.nn.l2_loss(b_soft)
            # scores = tf.matmul(drop_rnn, W_soft) + b_soft
            scores = tf.nn.xw_plus_b(x=drop_rnn, weights=W_soft, biases=b_soft, name='scores')
            predictions = tf.argmax(scores, 1, name='predictions')

        # Calculate loss function
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(scores, input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss / 2

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

