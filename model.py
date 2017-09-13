import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.python.layers import core as layers_core

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


def get_decoder_outputs(iterator, initial_state):
    num_units = 20

    with tf.variable_scope('embedding') as scope:
        embedding_decoder = tf.get_variable('embedding_decoder', [tgt_vocab_size, tgt_embedding_size], dtype)

        target = iterator.target

        if time_major: target = tf.transpose(target)

        decoder_emb_input = tf.nn.embedding_lookup(embedding_decoder, target)

    decoder_cell = rnn.BasicLSTMCell(num_units=num_units)

    helper = seq2seq.TrainingHelper(
        decoder_emb_input, iterator.target_length, time_major=True
    )

    projection_layer = layers_core.Dense(tgt_vocab_size, use_bias=False)

    decoder = seq2seq.BasicDecoder(
        decoder_cell, helper, initial_state, output_layer=projection_layer
    )

    time_axis = 0 if time_major else 1

    max_time = target

    target_weights = tf.sequence_mask(iterator.target_length, )
    outputs, _, _ = seq2seq.dynamic_decode(decoder)

    logits = outputs.rnn_output

    return logits


def loss(decoder_outputs, logits):
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=decoder_outputs, logits=logits
    )



if __name__ == '__main__':
    params = {
        'src_file': 'data_utils/source.txt',
        'tgt_file': 'data_utils/target.txt',
        'src_vocab_file': 'data_utils/source_vocab.txt',
        'tgt_vocab_file': 'data_utils/target_vocab.txt'
    }

    iterator = iterator_utils.get_iterator(**params)

    enc_outputs, enc_state = get_encoder_outputs(iterator)

    logits = get_decoder_outputs(iterator, enc_state)

    with tf.Session() as sess:
        iterator.initializer.run()
        tf.tables_initializer().run()
        tf.global_variables_initializer().run()

        outputs = sess.run(enc_outputs)
        print(outputs.shape)

        _logit = sess.run(logits)
        print(_logit.shape)
