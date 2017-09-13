from tensorflow.contrib.lookup import index_table_from_file

UNK = '<unk>'
SOS = '<s>'
EOS = '</s>'
UNK_ID = 0


def check_vocab(vocab_file, sos=None, eos=None, unk=None):
    with open(vocab_file, 'r') as f:
        vocab = []
        vocab_size = 0
        for word in f:
            vocab_size += 1
            vocab.append(word.strip())

    if not unk: unk = UNK
    if not sos: sos = SOS
    if not eos: eos = EOS

    if vocab[0:3] != [unk, sos, eos]:
        vocab = [unk, sos, eos] + vocab
        vocab_size += 3

        with open(vocab_file, 'w') as f:
            for word in vocab:
                f.write('%s\n' % word)

    return vocab_file, vocab_size


def create_vocab_tables(src_vocab_file, tgt_vocab_file):
    src_vocab_table = index_table_from_file(
        src_vocab_file, default_value=UNK_ID
    )

    tgt_vocab_table = index_table_from_file(
        tgt_vocab_file, default_value=UNK_ID
    )

    return src_vocab_table, tgt_vocab_table

