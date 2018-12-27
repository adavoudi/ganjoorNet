import tensorflow as tf
import numpy as np
import my_txtutils

# these must match what was saved !
ALPHASIZE = my_txtutils.ALPHASIZE
NLAYERS = 4
INTERNALSIZE = 1024

shahnameh = "./checkpoints/rnn_train_1492872774-1088500000"

# use topn=10 for all but the last which works with topn=2 for Shakespeare and topn=3 for Python
author = shahnameh
meta_graph = "./checkpoints/rnn_train_1492872774-1088500000.meta"

ncnt = 0
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(meta_graph)
    new_saver.restore(sess, author)

    file = open("sher.txt", "w")
    inputFile = open("test.txt", "r")

    init_text = inputFile.read().decode('utf8')
    encoded_text = my_txtutils.encode_text(init_text);
    # y = np.array([[encoded_text]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
    h = np.zeros([1, INTERNALSIZE * NLAYERS], dtype=np.float32)  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]
    for char in init_text:
        file.write(char.encode('utf8'));

    for i in range(len(encoded_text)-1):
        y = np.array([[encoded_text[i]]])
        yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0': 1})

    y = np.array([[encoded_text[-1]]])

    for i in range(50):
        yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0': 1})

        # If sampling is be done from the topn most likely characters, the generated text
        # is more credible and more "english". If topn is not set, it defaults to the full
        # distribution (ALPHASIZE)

        # Recommended: topn = 10 for intermediate checkpoints, topn=2 for fully trained checkpoints

        c = my_txtutils.sample_from_probabilities(yo, topn=1)
        y = np.array([[c]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
        # c = chr(my_txtutils.convert_to_alphabet(c))
        if(c == 37):
            continue

        c = chr(my_txtutils.convert_to_alphabet(c))
        print(c, end="")
        file.write(c.encode('utf8'))

        if c == '\n':
            ncnt = 0
        else:
            ncnt += 1
        if ncnt == 100:
            print("")
            ncnt = 0

    file.close()
