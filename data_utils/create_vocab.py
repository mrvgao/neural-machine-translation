source_vocab = open('source_vocab.txt', 'w')
start = ord('a')
words = '\n'.join([chr(i) for i in range(start, start + 2)])
source_vocab.write(words)

target_vocab = open('target_vocab.txt', 'w')
start = ord('A')
words = '\n'.join([chr(i) for i in range(start, start + 2)])
target_vocab.write(words)
