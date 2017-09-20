import collections

import numpy as np
import tensorflow as tf
from tensorflow.contrib import data as tf_data

from data_utils import table_utils

unk = '*'
sos = '.'
eos = '-'
unk_id = 0
sos_id = 1
eos_id = 2


source_dataset = tf_data.TextLineDataset('source.txt')
source_dataset = source_dataset.map(lambda string: tf.string_split([string].values))

with tf.Session() as sess:
    for i in range(5):
        s_t = sess.run(source_dataset)

        print(s_t)


