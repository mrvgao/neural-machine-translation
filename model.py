import os
from collections import namedtuple
from datetime import datetime

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.python.layers import core as layers_core

from data_utils import iterator_utils

os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5, 6'

src_vocab_size = 29
tgt_vocab_size = 29

src_embedding_size = 2
tgt_embedding_size = 2

dtype = tf.float32

time_major = True

Hyperpamamters = namedtuple('hps', ['learning_rate', 'batch_size',
                                    'max_gradient_norm', 'num_units',
                                    'attention', 'att_num_units',
                                    'stack_layers'])


class Model:
    def __init__(self, iterator, _hps):
        self.iterator = iterator
        self.time_major = True
        self.hps = _hps
        self.__build_embedding__()

        encoder_outputs, encoder_state = self.build_encode()
        logits = self.build_decode(encoder_outputs, encoder_state)

        self.label_hat_probabilities = tf.nn.softmax(logits)
        self.loss, self.summary = self.compute_loss(logits)
        self.update = self.optimize(self.loss)

    def __build_embedding__(self):
        with tf.variable_scope('embedding'):
            embedding_encoder = tf.get_variable(
                'embedding_encoder', [src_vocab_size, src_embedding_size], dtype)
            embedding_decoder = tf.get_variable(
                'embedding_decoder', [tgt_vocab_size, tgt_embedding_size], dtype)

        source = self.iterator.source
        target_input = self.iterator.target_input

        self.encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, source)
        self.decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, target_input)

        if self.time_major:
            self.encoder_emb_inp = tf.transpose(self.encoder_emb_inp, [1, 0, 2])
            self.decoder_emb_inp = tf.transpose(self.decoder_emb_inp, [1, 0, 2])

    def build_encode(self):
        # encoder_cell = rnn.BasicLSTMCell(num_units=self.hps.num_units)
        encoder_cell = rnn.GRUCell(num_units=self.hps.num_units)

        with tf.variable_scope('dynamic_seq2seq', dtype=dtype) as scope:
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                cell=encoder_cell, inputs=self.encoder_emb_inp,
                sequence_length=self.iterator.source_length,
                time_major=True,
                dtype=dtype
            )

        return encoder_outputs, encoder_state

    def build_decoder_cell(self, encoder_state):
        # decoder_cell = rnn.BasicLSTMCell(num_units=self.hps.num_units)
        decoder_cell = rnn.GRUCell(num_units=self.hps.num_units)
        decoder_initial_state = encoder_state

        return decoder_cell, decoder_initial_state

    def build_attention_cell(self, encoder_outputs, encoder_states):
        memory = encoder_outputs

        if self.time_major:
            memory = tf.transpose(encoder_outputs, [1, 0, 2])

        attention_mechanism = seq2seq.BahdanauAttention(
            num_units=self.hps.att_num_units, memory=memory,
            memory_sequence_length=self.iterator.target_length
        )

        cell = rnn.BasicLSTMCell(num_units=self.hps.num_units)

        # cell = rnn.MultiRNNCell([cell] * self.hps.stack_layers)

        cell = seq2seq.AttentionWrapper(
            cell, attention_mechanism,
            attention_layer_size=self.hps.att_num_units, name='attention'
        )

        batch_size = tf.size(self.iterator.source_length)

        decoder_initial_state = cell.zero_state(batch_size=batch_size, dtype=dtype).clone(
            cell_state=encoder_states
        )

        return cell, decoder_initial_state

    def build_decode(self, encoder_outputs, encoder_state):
        if self.hps.attention:
            decoder_cell, decoder_initial_state = self.build_attention_cell(encoder_outputs, encoder_state)
        else:
            decoder_cell, decoder_initial_state = self.build_decoder_cell(encoder_state)

        helper = seq2seq.TrainingHelper(
            self.decoder_emb_inp, self.iterator.target_length, time_major=True
        )
        projection_layer = layers_core.Dense(tgt_vocab_size, use_bias=False)

        decoder = seq2seq.BasicDecoder(
            decoder_cell, helper, decoder_initial_state, output_layer=projection_layer
        )

        outputs, _, _ = seq2seq.dynamic_decode(decoder)

        logits = outputs.rnn_output

        return logits

    def compute_loss(self, logits):
        target_output = self.iterator.target_output

        if self.time_major:
            target_output = tf.transpose(target_output)

        max_time = self.get_max_time(target_output)
        target_weights = tf.sequence_mask(lengths=self.iterator.target_length,
                                          maxlen=max_time)

        if self.time_major:
            target_weights = tf.transpose(target_weights)

        target_weights = tf.cast(target_weights, tf.float32)

        # crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=logits)

        # _loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(self.hps.batch_size)
        _loss = seq2seq.sequence_loss(logits=logits, targets=target_output,
                                      weights=target_weights)

        tf.summary.scalar(name='seq2seq-loss', tensor=_loss)

        summary_merged = tf.summary.merge_all()

        return _loss, summary_merged

    def optimize(self, loss):
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.hps.max_gradient_norm)

        global_step = tf.Variable(0, trainable=True)
        starter_learning_rate = self.hps.learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   1000, 0.9, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        op = optimizer.apply_gradients(zip(clipped_gradients, params))

        return op

    def get_max_time(self, tensor):
        time_axis = 0 if self.time_major else 1
        return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

    def train(self, sess):
        return sess.run([
            self.update,
            self.loss,
            self.summary,
            self.label_hat_probabilities,
        ])


if __name__ == '__main__':
    hps = Hyperpamamters(
        learning_rate=1e-2,
        batch_size=1024,
        max_gradient_norm=2,
        num_units=128,
        attention=False,
        att_num_units=64,
        stack_layers=2,
    )

    params = {
        'src_file': 'data_utils/source.txt',
        'tgt_file': 'data_utils/target.txt',
        'src_vocab_file': 'data_utils/source_vocab.txt',
        'tgt_vocab_file': 'data_utils/target_vocab.txt',
        'batch_size': hps.batch_size
    }

    _iterator = iterator_utils.get_iterator(**params)

    seq2seq_model = Model(iterator=_iterator, _hps=hps)

    num_epoch = 10
    epoch = 0

    train_session = tf.Session()

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    logdir = 'tf-log'

    summary_writer = tf.summary.FileWriter("{}/run-{}".format(logdir, now))

    train_session.run(tf.tables_initializer())
    train_session.run(tf.global_variables_initializer())

    epoch = 0
    max_epoch = 100
    total_steps = 0

    saver = tf.train.Saver()

    with train_session:
        tf.tables_initializer().run()
        tf.global_variables_initializer().run()
        seq2seq_model.iterator.initializer.run()

        while epoch < max_epoch:
            print('epoch -- {} --- epoch'.format(epoch))
            try:
                _, loss, summary, label_hats = seq2seq_model.train(sess=train_session)
                print("total_steps: {} loss: {}".format(total_steps, loss))
                total_steps += 1
                summary_writer.add_summary(summary, global_step=total_steps)

                if total_steps % 10 == 0:
                    summary_writer.flush()

                if total_steps % 1000 == 0:
                    saver.save(train_session, 'models/neural-translation-with-loss-{}-steps-{}'.format(loss, total_steps),
                               global_step=total_steps)
            except tf.errors.OutOfRangeError:
                epoch += 1
                train_session.run(seq2seq_model.iterator.initializer)
                continue

