from char_level_cnn_rnn.preprocess import quantization
from char_level_cnn_rnn.models import TextCNNRNN
import tensorflow as tf
import numpy as np


def load_raw(typ):
    col_list = []
    with open('%s/ratings_%s.txt' % (file_dir, typ), 'r') as f:
        for n, line in enumerate(f.readlines()[1:]):
            try:
                index, document, label = line.split('\t')
            except IndexError:
                continue
            col_list.append({'label': [label],
                             'embedded': qua.sent_quantize(document)})
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

sequence_length = 336
num_classes = 2
vocab_size = 111
embedding_size = 8
num_filter = 128
filter_lengths = [5, 3]
pool_lengths = [2, 2]
dropout = 0.5
hidden_size = 128
l2_reg_lambda = 5e-4

init = tf.initialize_all_variables()
qua = quantization.Quantization()
file_dir = '/Users/Hanseok/PycharmProjects/mldl-sandbox/char_level_cnn_rnn/source'

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        model = TextCNNRNN(sequence_length=sequence_length, num_classes=num_classes, vocab_size=vocab_size,
                           embedding_size=embedding_size, num_filter=num_filter, filter_lengths=filter_lengths,
                           pool_lengths=pool_lengths, dropout=dropout, hidden_size=hidden_size,
                           l2_reg_lambda=l2_reg_lambda)
        optimizer = tf.train.AdadeltaOptimizer(rho=0.95, epsilon=1e-5)

        loss_summ = tf.scalar_summary('loss', model.loss)
        acc_summ = tf.scalar_summary('acc', model.accuracy)
        merged = tf.merge_all_summaries()

        batch_size = 10000
        istate = np.zeros((batch_size, 2*hidden_size))

        for batch in batch_loader(batch_size):
            batch_x, batch_y = batch
            feed_dict = {'input_x': batch_x, 'input_y': batch_y, 'istate_fw': istate, 'istate_bw': istate}
            sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
