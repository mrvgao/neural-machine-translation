import collections

import numpy as np
import tensorflow as tf
from tensorflow.contrib import data as tf_data

from data_utils import table_utils

unk = '*'
sos = '.'
eos = '-'
unk_id = 0
sos_id = 1
eos_id = 2

BatchedInput = collections.namedtuple(
    'BatchedInput', ['target_input', 'target_length', 'source', 'source_length',
                     'initializer', 'target_output'])


def get_iterator(src_file, tgt_file, src_vocab_file, tgt_vocab_file, batch_size):
    source_dataset = tf_data.TextLineDataset(src_file)
    target_dataset = tf_data.TextLineDataset(tgt_file)

    src_file, src_size = table_utils.check_vocab(src_vocab_file, eos=eos, unk=unk, sos=sos)
    tgt_file, tgt_size = table_utils.check_vocab(tgt_vocab_file, eos=eos, unk=unk, sos=sos)

    src_table, tgt_table = table_utils.create_vocab_tables(src_vocab_file=src_file, tgt_vocab_file=tgt_file)

    source_dataset = source_dataset.map(lambda string: tf.string_split([string]).values)
    source_dataset = source_dataset.map(lambda words,: tf.cast(src_table.lookup(words), tf.int32))

    target_dataset = target_dataset.map(lambda string: tf.string_split([string]).values)
    target_dataset = target_dataset.map(lambda words: tf.cast(tgt_table.lookup(words), tf.int32))

    source_target_dataset = tf_data.Dataset.zip((source_dataset, target_dataset))

    # TODO<minquan>: Add similar size batched.

    source_target_dataset = source_target_dataset.map(
        lambda src, tgt: (src, tgt, tgt),
        num_threads=8
    )

    source_target_dataset = source_target_dataset.map(
        lambda src, tgt_in, tgt_out: (src,
                                      tf.concat(([sos_id], tgt_in), axis=0),
                                      tf.concat((tgt_out, [eos_id]), axis=0),
                                      tf.size(src), tf.size(tgt_in) + 1)
    )

    source_target_dataset = source_target_dataset.shuffle(buffer_size=100000)

    batched_dataset = source_target_dataset.padded_batch(
        batch_size,
        padded_shapes=(
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([]),
            tf.TensorShape([]),
        ),

        padding_values=(
            eos_id,
            eos_id,
            eos_id,
            0,
            0)
    )

    batched_iterator = batched_dataset.make_initializable_iterator()

    source, target_input, target_output, source_length, target_length = batched_iterator.get_next()

    return BatchedInput(initializer=batched_iterator.initializer,
                        source=source,
                        source_length=source_length,
                        target_input=target_input,
                        target_output=target_output,
                        target_length=target_length)


def iterator_mock():
    batch_input = get_iterator(
        src_file='source.txt', src_vocab_file='source_vocab.txt',
        tgt_file='target.txt', tgt_vocab_file='target_vocab.txt',
        batch_size=128,
    )

    with tf.Session() as sess:
        tf.tables_initializer().run()
        batch_input.initializer.run()
        src, src_length = sess.run([batch_input.source, batch_input.source_length])
        source = sess.run([batch_input.source])
        target_input = sess.run([batch_input.target_input])
        target_output = sess.run([batch_input.target_output])

        print('source shape is {}'.format(np.array(source).shape))
        print('target_input shape is {}'.format(np.array(target_input).shape))
        print('target_output shape is {}'.format(np.array(target_output).shape))

        assert np.max(target_input) <= 25 + 3, np.max(target_input)
        assert np.max(source) <= 25 + 3, np.max(source)
        assert np.max(target_output) <= 25 + 3, np.max(target_output)

        print(src[:10])
        print(src_length[:10])

        print('test done!')


def lookup_mock():
    src_table, tgt_table = table_utils.create_vocab_tables(src_vocab_file='source_vocab.txt',
                                                           tgt_vocab_file='target_vocab.txt')

    words = tf.constant(['.', '*', 'a', 'X', 'Z', 'Z'], dtype=tf.string)

    table = tf.contrib.lookup.index_to_string_table_from_file('target_vocab.txt', default_value='unknow')
    indices = tf.constant([0, 1, 28, 29], tf.int64)

    with tf.Session() as sess:
        tf.tables_initializer().run()
        word_id = sess.run(table.lookup(indices))
        print(word_id)


if __name__ == '__main__':
    iterator_mock()
