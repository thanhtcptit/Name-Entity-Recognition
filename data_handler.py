def get_sentence(file):
    data = []
    sentence = []
    with open(file, 'r') as f:
        for line in f:
            if line in ['\n', '\r\n']:
                if len(sentence) == 0:
                    continue
                else:
                    data.append(sentence)
                    sentence = []
            else:
                sentence.append(line)
    return data

def validate_data(file, org_file):
    data = get_sentence(file)
    org_data = get_sentence(org_file)

    print (file + "- Number of sentence: ", len(data))
    print (org_file + "- Number of sentence: ", len(org_data))
    print (file + "- Number of tokens: ", sum([len(i) for i in data]))
    print (org_file + "- Number of tokens: ", sum([len(i) for i in data]))
    diff_count = 0
    for line, org_line in zip(data, org_data):
        assert len(line) == len(org_line)
        for word, org_word in zip(line, org_line):
            tokens = word.split(' ')
            org_tokens = org_word.split(' ')
            if tokens[0] != org_tokens[0]:
                diff_count += 1
                print ('word: {} - org word: {}'.format(word, org_word))
    print ('Total different tokens: ', diff_count)

def NER_to_sentence(file):
    max_sentence_len = 30
    sentences = []
    sentences_tag = []
    sentence = []
    sentence_tag = []
    with open(file, 'r') as f:
        for line in f:
            if line in ['\n', '\r\n']:
                if len(sentence) == 0:
                    continue
                if len(sentence) > max_sentence_len:
                    sentence = []
                    sentence_tag = []
                    continue
                padding = max_sentence_len - len(sentence)
                for i in range(padding):
                    sentence.append('PAD')
                    sentence_tag.append('PAD')
                sentences.append(['PAD'] * 2 + sentence + ['PAD'] * 2)
                sentences_tag.append(sentence_tag)
                sentence = []
                sentence_tag = []
            else:
                tokens = line.split(' ')
                assert len(tokens) == 4
                sentence.append(tokens[0])
                sentence_tag.append(tokens[3].strip())
    with open('data/train_sentences.txt', 'w') as f:
        for s in sentences:
            f.write(' '.join(s) + '\n')
    with open('data/tags.txt', 'w') as f:
        for s in sentences_tag:
            f.write(' '.join(s) + '\n')

def POS_to_sentence(file):
    max_sentence_len = 30
    data = []
    tags = []
    train_percent = 0.8
    with open(file, 'r') as f:
        for line in f:
            words = line.strip().split(' ')
            if (len(words) > max_sentence_len):
                continue
            sentence = []
            tag = []
            for word in words:
                split = word.find('/')
                if (word[split + 1:].find('/') != -1):
                    continue
                if split == -1:
                    continue
                sentence.append(word[:split])
                tag.append(word[split + 1:])
            pad_len = max_sentence_len - len(words)
            data.append(sentence + ['PAD'] * pad_len)
            tags.append(tag + ['PAD'] * pad_len)
    train_bound = int(len(data) * train_percent)
    print ("Train size: ", train_bound)
    with open('data/pos/train.txt', 'w') as f:
        for s in data[:train_bound]:
            f.write(' '.join(s) + '\n')
    with open('data/pos/tags.txt', 'w') as f:
        for t in tags[:train_bound]:
            f.write(' '.join(t) + '\n')

def build_vocab(file):
    vocab = []
    vocab.append('PAD')
    vocab.append('<UNK>')
    with open(file, 'r') as f:
        data = f.read().split('\n')
    for line in data:
        words = line.split(' ')
        for w in words:
            if w in vocab:
                continue
            else:
                vocab.append(w)
    with open('data/sample/vocab.txt', 'w') as f:
        for w in vocab:
            f.write(w + '\n')

def tag_vocab(file):
    vocab = []
    with open(file, 'r') as f:
        data = f.read().split('\n')
    for line in data:
        words = line.split(' ')
        for w in words:
            if w in vocab:
                continue
            else:
                vocab.append(w)
    with open('data/pos/tags_vocab.txt', 'w') as f:
        for w in vocab:
            f.write(w + '\n')

if __name__ == '__main__':
    build_vocab()
