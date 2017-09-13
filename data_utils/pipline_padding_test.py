import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator


eos = '-1'
padding = '0'

string1 = 'a b a d e f a g e a d g g h r e f g j'
string2 = 'a g r c g h r h r h r f j k a e'

all_words = set(string1.split() + string2.split())
all_words = list(all_words) + [eos, padding]


test_batch = ['a b c c c', 'd e c', 'd e c c',
              'd e e', 'd e e e e e', 'd']

test_batch = sorted(test_batch, key=lambda x: len(x))
src_strings = tf.placeholder(shape=[None], dtype=tf.string, name='source')

src_dataset = Dataset.from_tensor_slices(src_strings)

src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)
src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))

batch_size = 2

batched_dataset = src_dataset.padded_batch(
    batch_size,
    padded_shapes=(
        tf.TensorShape([None]),
        tf.TensorShape([])
    ),
    padding_values=(eos, 0)
)

batch_iterator = batched_dataset.make_initializable_iterator()
next_element = batch_iterator.get_next()

with tf.Session() as sess:
    sess.run(batch_iterator.initializer, feed_dict={src_strings: test_batch})
    while True:
        try:
            string, length = sess.run(next_element)
            print(string, length)
        except tf.errors.OutOfRangeError:
            break

