import os
from collections import namedtuple

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.python.layers import core as layers_core

from data_utils import iterator_utils

os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5, 6'

src_vocab_size = 29
tgt_vocab_size = 15

src_embedding_size = 10
tgt_embedding_size = 8

dtype = tf.float32

time_major = True

Hyperpamamters = namedtuple('hps', ['learning_rate', 'batch_size',
                                    'max_gradient_norm', 'num_units',
                                    'attention'])


class Model:
    def __init__(self, iterator, hps):
        self.iterator = iterator
        self.time_major = True
        self.hps = hps
        self.__build_embedding__()

    def __build_embedding__(self):
        with tf.variable_scope('embedding') as scope:
            embedding_encoder = tf.get_variable('embedding_encoder', [src_vocab_size, src_embedding_size], dtype)
            embedding_decoder = tf.get_variable('embedding_decoder', [tgt_vocab_size, tgt_embedding_size], dtype)

        source = iterator.source
        target_input = iterator.target_input

        if self.time_major:
            source = tf.transpose(source)
            target_input = tf.transpose(target_input)

        self.encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, source)
        self.decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, target_input)

    def encode(self):
        encoder_cell = rnn.BasicLSTMCell(num_units=self.hps.num_units)

        with tf.variable_scope('dynamic_seq2seq', dtype=dtype) as scope:
            self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                cell=encoder_cell, inputs=self.encoder_emb_inp,
                sequence_length=iterator.source_length,
                time_major=True,
                dtype=dtype
            )

    def decode(self):
        attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])

        attention_mechanism = seq2seq.LuongAttention(
            num_units=self.hps.num_units, memory=attention_states,
            memory_sequence_length=self.iterator.source_length
        )

        decoder_cell = rnn.BasicLSTMCell(num_units=self.hps.num_units)

        decoder_cell = seq2seq.AttentionWrapper(
            decoder_cell, attention_mechanism,
            attention_layer_size=self.hps.num_units,
            name='attention'
        )

        decoder_initial_state = decoder_cell.zero_state(batch_size=self.hps.batch_size, dtype=dtype).clone(
            cell_state=self.encoder_state
        )

        helper = seq2seq.TrainingHelper(
            self.decoder_emb_inp, iterator.target_length, time_major=True
        )

        projection_layer = layers_core.Dense(tgt_vocab_size, use_bias=False)

        decoder = seq2seq.BasicDecoder(
            decoder_cell, helper, decoder_initial_state, output_layer=projection_layer
        )

        outputs, _, _ = seq2seq.dynamic_decode(decoder)

        logits = outputs.rnn_output

        self.logits = logits

    def compute_loss(self):
        target_output = self.iterator.target_output

        if self.time_major:
            target_output = tf.transpose(target_output)

        max_time = self.get_max_time(target_output)
        target_weights = tf.sequence_mask(lengths=self.iterator.target_length,
                                          maxlen=max_time)

        if self.time_major:
            target_weights = tf.transpose(target_weights)

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=self.logits)

        target_weights = tf.cast(target_weights, tf.float32)

        self.loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(self.hps.batch_size)

    def optimize(self):
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.hps.max_gradient_norm)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.hps.learning_rate)

        self.update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

    def get_logits(self):
        self.encode()
        self.decode()

    def train_batch(self):
        self.compute_loss()
        self.optimize()

    def get_max_time(self, tensor):
        time_axis = 0 if self.time_major else 1
        return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]


if __name__ == '__main__':
    hps = Hyperpamamters(
        learning_rate=1e-3,
        batch_size=128,
        max_gradient_norm=5,
        num_units=50,
        attention=True
    )

    params = {
        'src_file': 'data_utils/source.txt',
        'tgt_file': 'data_utils/target.txt',
        'src_vocab_file': 'data_utils/source_vocab.txt',
        'tgt_vocab_file': 'data_utils/target_vocab.txt',
        'batch_size': hps.batch_size
    }

    iterator = iterator_utils.get_iterator(**params)

    seq2seq_model = Model(iterator=iterator, hps=hps)
    seq2seq_model.get_logits()
    seq2seq_model.compute_loss()
    seq2seq_model.optimize()

    num_epoch = 10
    with tf.Session() as sess:
        tf.tables_initializer().run()
        tf.global_variables_initializer().run()

        for epoch in range(num_epoch):
            iterator.initializer.run()

            print('epoch ---- {} ---- epoch'.format(epoch))
            index = 0
            while True:
                try:
                    loss, _ = sess.run([seq2seq_model.loss, seq2seq_model.update_step])
                    if index % 50 == 0: print('epoch: {}, loss: {}'.format(epoch, loss))
                    index += 1
                except tf.errors.OutOfRangeError:
                    break
