SAVE_DIR = 'saved_model/tmp1/'

SENTENCE_LEN = 10
WORD_DIM = 111
NUM_CLASSES = 9

NUM_RNN_LAYERS = 2
HIDDEN_SIZE = 128
OUTPUT_SIZE = 2 * HIDDEN_SIZE
KEEP_PR = 0.5
WINDOW_SIZE = 2

def write_config():
    with open(SAVE_DIR + 'config.txt', 'w') as f:
        f.write(str(SENTENCE_LEN) + '\n')
        f.write(str(WORD_DIM) + '\n')
        f.write(str(NUM_RNN_LAYERS) + '\n')
        f.write(str(HIDDEN_SIZE) + '\n')
        f.write(str(KEEP_PR) + '\n')
        f.write(str(WINDOW_SIZE) + '\n')

write_config()