import tensorflow as tf
from tensorflow.contrib import rnn

from data_utils import iterator_utils

src_vocab_size = 29
tgt_vocab_size = 13

src_embedding_size = 10
tgt_embedding_size = 8

dtype = tf.float32

time_major = True


def get_encoder_outputs(iterator):
    with tf.variable_scope('embedding') as scope:
        embedding_encoder = tf.get_variable('embedding_encoder', [src_vocab_size, src_embedding_size], dtype)

        source = iterator.source

        if time_major:
            source = tf.transpose(source)

        encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, source)

    num_units = 20
    encoder_cell = rnn.BasicLSTMCell(num_units=num_units)

    with tf.variable_scope('dynamic_seq2seq', dtype=dtype) as scope:
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            cell=encoder_cell, inputs=encoder_emb_inp,
            sequence_length=iterator.source_length,
            time_major=True,
            dtype=dtype
        )

    return encoder_outputs, encoder_state


if __name__ == '__main__':
    params = {
        'src_file': 'data_utils/source.txt',
        'tgt_file': 'data_utils/target.txt',
        'src_vocab_file': 'data_utils/source_vocab.txt',
        'tgt_vocab_file': 'data_utils/target_vocab.txt'
    }

    iterator = iterator_utils.get_iterator(**params)

    enc_outputs, enc_state = get_encoder_outputs(iterator)

    with tf.Session() as sess:
        iterator.initializer.run()
        tf.tables_initializer().run()
        tf.global_variables_initializer().run()

        outputs = sess.run(enc_outputs)
        print(outputs.shape)
