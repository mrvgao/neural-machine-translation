source_vocab = open('source_vocab.txt', 'w')
start = ord('a')
words = '\n'.join([chr(i) for i in range(start, start + 26)])
source_vocab.write(words)

target_vocab = open('target_vocab.txt', 'w')
start = ord('0')
words = '\n'.join([chr(i) for i in range(start, start + 10)])
target_vocab.write(words)
