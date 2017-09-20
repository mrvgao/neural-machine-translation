from data_utils import iterator_utils
import tensorflow as tf


params = {
    'src_file': 'data_utils/source.txt',
    'tgt_file': 'data_utils/target.txt',
    'src_vocab_file': 'data_utils/source_vocab.txt',
    'tgt_vocab_file': 'data_utils/target_vocab.txt',
    'batch_size': 2
}


class IteratorMock:
    def __init__(self, iterator):
        self.iterator = iterator
        self.source = self.get_source()
        self.target = self.get_target(self.source)

    def get_source(self):
        return self.iterator.source

    def get_target(self, source):
        return tf.concat([self.iterator.target_output, source], axis=1)

    def train(self, sess):
        return sess.run([self.source, self.target])


iterator = iterator_utils.get_iterator(**params)
model = IteratorMock(iterator)

with tf.Session() as sess:
    iterator.initializer.run()
    tf.tables_initializer().run()

    for i in range(2):
        source, target = model.train(sess)
        print(source)
        print(target)
        print('batch -- {} ------batch --'.format(i))

