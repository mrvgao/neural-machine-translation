import collections

import tensorflow as tf
from tensorflow.contrib import data as tf_data

from data_utils import table_utils

unk = '*'
sos = '.'
eos = '-'
eos_id = 1

BatchedInput = collections.namedtuple(
    'BatchedInput', ['target', 'target_length', 'source', 'source_length', 'initializer'])


def get_iterator(src_file, tgt_file, src_vocab_file, tgt_vocab_file):
    source_dataset = tf_data.TextLineDataset(src_file)
    target_dataset = tf_data.TextLineDataset(tgt_file)

    src_file, src_size = table_utils.check_vocab(src_vocab_file, eos=eos, unk=unk, sos=sos)
    tgt_file, tgt_size = table_utils.check_vocab(tgt_vocab_file, eos=eos, unk=unk, sos=sos)

    src_table, tgt_table = table_utils.create_vocab_tables(src_vocab_file=src_file, tgt_vocab_file=tgt_file)

    source_dataset = source_dataset.map(lambda string: tf.string_split([string]).values)
    source_dataset = source_dataset.map(lambda words: (words, tf.size(words)))
    source_dataset = source_dataset.map(lambda words, size: (src_table.lookup(words), size))

    target_dataset = target_dataset.map(lambda string: tf.string_split([string]).values)
    target_dataset = target_dataset.map(lambda words: (words, tf.size(words)))
    target_dataset = target_dataset.map(lambda words, size: (tgt_table.lookup(words), size))

    source_target_dataset = tf_data.Dataset.zip((source_dataset, target_dataset))

    # TODO<minquan>: Add similar size batched.

    batch_size = 256

    batched_dataset = source_target_dataset.padded_batch(
        batch_size,
        padded_shapes=((tf.TensorShape([None]), tf.TensorShape([])),
                       (tf.TensorShape([None]), tf.TensorShape([]))),
        padding_values=((tf.cast(eos_id, tf.int64), 0),
                        (tf.cast(eos_id, tf.int64), 0))
    )

    batched_iterator = batched_dataset.make_initializable_iterator()

    ((source, source_length), (target, target_length)) = batched_iterator.get_next()

    return BatchedInput(initializer=batched_iterator.initializer,
                        source=source,
                        source_length=source_length,
                        target=target,
                        target_length=target_length)


if __name__ == '__main__':
    batch_input = get_iterator(
        src_file='source.txt', src_vocab_file='source_vocab.txt',
        tgt_file='target.txt', tgt_vocab_file='target_vocab.txt'
    )

    with tf.Session() as sess:
        tf.tables_initializer().run()
        batch_input.initializer.run()
        src, src_length = sess.run([batch_input.source, batch_input.source_length])

        print(src[:10])
        print(src_length[:10])
