import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from data_utils import iterator_utils


class SingleRNN:
    def __init__(self, iterator):
        num_units = 10
        dtype = tf.float32

        self.iterator = iterator
        self.input = tf.cast(iterator.source_each_timestep, dtype=dtype)
        self.output = tf.cast(iterator.target_each_timestep, dtype=dtype)

        time_major = False

        with tf.variable_scope('rnn') as scope:
            rnn_cell = rnn.MultiRNNCell(
                [rnn.BasicLSTMCell(num_units=num_units, state_is_tuple=True)]
            )

            with tf.variable_scope('dynamic_rnn', dtype=dtype) as scope:
                rnn_outputs, rnn_state = tf.nn.dynamic_rnn(
                    cell=rnn_cell, inputs=self.input,
                    sequence_length=iterator.source_length,
                    dtype=dtype
                )

            num_outputs = 1

            stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, num_units])

            projector_outputs = layers.fully_connected(inputs=stacked_rnn_outputs,
                                                       num_outputs=num_outputs)

            outputs = tf.reshape(projector_outputs, [-1, 5, num_outputs])

            self.loss = tf.losses.mean_squared_error(labels=self.output, predictions=outputs)
            self.op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss)

    def train(self, sess):
        return sess.run([self.loss, self.op])


def main(_):
    params = {
        'src_file': 'data_utils/source.txt',
        'tgt_file': 'data_utils/target.txt',
        'src_vocab_file': 'data_utils/source_vocab.txt',
        'tgt_vocab_file': 'data_utils/target_vocab.txt',
        'batch_size': 256
    }

    iterator= iterator_utils.get_iterator(**params)

    model = SingleRNN(iterator=iterator)

    train_session = tf.Session()

    with train_session:
        train_session.run(tf.global_variables_initializer())
        train_session.run(tf.tables_initializer())
        model.iterator.initializer.run()

        while True:
            try:
                loss, _ = model.train(sess=train_session)
                print('loss: {}'.format(loss))
            except tf.errors.OutOfRangeError:
                print('end')


if __name__ == '__main__':
    tf.app.run()
