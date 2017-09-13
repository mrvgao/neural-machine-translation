import tensorflow as tf

from tensorflow.contrib.data import Dataset, Iterator


NUM_ClASS = 2


def input_parser(img_path, label):
    one_hot = tf.one_hot(label, NUM_ClASS)
    # img_file = tf.read_file(img_path)
    # img_decoded = tf.image.decode_image(img_file, channels=3)
    img_decoded = img_path

    return img_decoded, one_hot


train_imgs = tf.constant(['train/img1.jpg', 'train/img2.jpg', 'train/img3.jpg',
                         'train/img4.jpg', 'train/img5.jpg', 'train/img6.jpg'
                         ])

train_labels = tf.constant([0, 0, 0, 1, 1, 1])

val_imgs = tf.constant(['val/img1.jpg', 'val/img2.jpg', 'val/img3.jpg'])

val_labels = tf.constant([0, 0, 1])

tr_dataset = Dataset.from_tensor_slices((train_imgs, train_labels))
val_dataset = Dataset.from_tensor_slices((val_imgs, val_labels))

# iterator = Iterator.from_structure(tr_dataset.output_types, tr_dataset.output_shapes)

tr_dataset = tr_dataset.map(input_parser)
val_dataset = val_dataset.map(input_parser)

batch_size = 2
# dataset = tr_dataset.batch(batch_size)

batched_dataset = tr_dataset.batch(batch_size)

iterator = Iterator.from_dataset(batched_dataset)

next_elements = iterator.get_next()

training_init_op = iterator.make_initializer(batched_dataset)
# validation_init_op = iterator.make_initializer(val_dataset)

with tf.Session() as sess:
    sess.run(training_init_op)
    while True:
        try:
            (source, label) = sess.run(next_elements)
            print(source)
            print(label)
        except tf.errors.OutOfRangeError:
            break




