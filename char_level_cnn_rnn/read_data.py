from preprocess import quantization as qt
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