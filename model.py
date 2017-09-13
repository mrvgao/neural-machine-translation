import tensorflow as tf

src_vocab_size = 29
tgt_vocab_size = 13

src_embedding_size = 10
tgt_embedding_size = 8

dtype = tf.float32


def get_encoder_outputs(iterator):
    with tf.variable_scope('embedding') as scope:
        embedding_encoder = tf.get_variable('embedding_encoder',
                                            [src_vocab_size, src_embedding_size],
                                            dtype)

        embedding_decoder = tf.get_variable('embedding_decoder',
                                            [tgt_vocab_size, tgt_embedding_size],
                                            dtype)
