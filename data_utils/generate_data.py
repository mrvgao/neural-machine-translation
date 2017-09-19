"""
Generate the training data.
"""

import random


def generate_char():
    minval = '0'
    maxval = '1'
    value = random.randint(ord(minval), ord(maxval))
    return chr(value), chr(value).upper()


def generate_seq(length=10):
    source_seq = ''
    target_seq = ''

    for ii in range(length):
        s, t = generate_char()
        source_seq += s
        target_seq += str(t)

    return ' '.join(source_seq), ' '.join(target_seq)[::-1]


def generate_batches_seq(batch_size=100000):
    source = open('source.txt', 'w')
    target = open('target.txt', 'w')
    for b in range(batch_size):
        length = random.randint(1, 5)
        s, t = generate_seq(length)
        source.write(s + '\n')
        target.write(t + '\n')

        if b % 100 == 0: print(b)

if __name__ == '__main__':
    print(generate_char())
    print(generate_seq(10))

    generate_batches_seq(100000)
