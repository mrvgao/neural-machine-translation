import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.python.layers import core as layers_core

from data_utils import iterator_utils

src_vocab_size = 29
tgt_vocab_size = 14

src_embedding_size = 10
tgt_embedding_size = 8

dtype = tf.float32

time_major = True


class Model:
    def __init__(self, iterator):
        self.iterator = iterator
        self.num_units = 20
        self.__build_embedding__()

    def get_logits(self):
        self.encode()
        self.decode()

    def __build_embedding__(self):
        with tf.variable_scope('embedding') as scope:
            embedding_encoder = tf.get_variable('embedding_encoder', [src_vocab_size, src_embedding_size], dtype)
            embedding_decoder = tf.get_variable('embedding_decoder', [tgt_vocab_size, tgt_embedding_size], dtype)

        source = iterator.source
        target_input = iterator.target_input

        if time_major:
            source = tf.transpose(source)
            target_input = tf.transpose(target_input)

        self.encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, source)
        self.decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, target_input)

    def encode(self):
        encoder_cell = rnn.BasicLSTMCell(num_units=self.num_units)

        with tf.variable_scope('dynamic_seq2seq', dtype=dtype) as scope:
            self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                cell=encoder_cell, inputs=self.encoder_emb_inp,
                sequence_length=iterator.source_length,
                time_major=True,
                dtype=dtype
            )

    def decode(self):
        decoder_cell = rnn.BasicLSTMCell(num_units=self.num_units)
        helper = seq2seq.TrainingHelper(
            self.decoder_emb_inp, iterator.target_length, time_major=True
        )

        projection_layer = layers_core.Dense(tgt_vocab_size, use_bias=False)

        decoder = seq2seq.BasicDecoder(
            decoder_cell, helper, self.encoder_state, output_layer=projection_layer
        )

        outputs, _, _ = seq2seq.dynamic_decode(decoder)

        logits = outputs.rnn_output

        self.logits = logits


if __name__ == '__main__':
    params = {
        'src_file': 'data_utils/source.txt',
        'tgt_file': 'data_utils/target.txt',
        'src_vocab_file': 'data_utils/source_vocab.txt',
        'tgt_vocab_file': 'data_utils/target_vocab.txt'
    }

    iterator = iterator_utils.get_iterator(**params)

    seq2seq_model = Model(iterator=iterator)
    seq2seq_model.get_logits()

    with tf.Session() as sess:
        iterator.initializer.run()
        tf.tables_initializer().run()
        tf.global_variables_initializer().run()

        _logit = sess.run(seq2seq_model.logits)
        print(_logit.shape)
