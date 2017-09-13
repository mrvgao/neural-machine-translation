import tensorflow as tf
from tensorflow.contrib import data as tf_data
from data_utils import table_utils

source_eos = '.'
target_eos = '-'

source_dataset = tf_data.TextLineDataset('source.txt')
target_dataset = tf_data.TextLineDataset('target.txt')

src_file, src_size = table_utils.check_vocab('source.txt', eos=source_eos)
tgt_file, tgt_size = table_utils.check_vocab('target.txt', eos=target_eos)

src_table, tgt_table = table_utils.create_vocab_tables(src_vocab_file=src_file, tgt_vocab_file=tgt_file)

source_dataset = source_dataset.map(lambda string: tf.string_split([string]).values)
source_dataset = source_dataset.map(lambda words: (words, tf.size(words)))
source_dataset = source_dataset.map(lambda words, size: (src_table.lookup(words), size))

target_dataset = target_dataset.map(lambda string: tf.string_split([string]).values)
target_dataset = target_dataset.map(lambda words: (words, tf.size(words)))
target_dataset = target_dataset.map(lambda words, size: (tgt_table.lookup(words), size))

source_target_dataset = tf_data.Dataset.zip((source_dataset, target_dataset))

batch_size = 256

batched_dataset = source_target_dataset.padded_batch(
    batch_size,
    padded_shapes=((tf.TensorShape([None]), tf.TensorShape([])), (tf.TensorShape([None])), tf.TensorShape([])),
    padding_values=((source_eos, 0), (target_eos, 0))
)

batched_iterator = batched_dataset.make_initializable_iterator()

((source, source_length), (target, target_length)) = batched_iterator.get_next()

with tf.Session() as sess:
    sess.run(batched_iterator.initializer)
