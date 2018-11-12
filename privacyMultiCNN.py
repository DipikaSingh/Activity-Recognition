import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from datetime import datetime


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys

from tensorflow.python.client import device_lib
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import embedding_rnn_decoder,rnn_decoder

# import pickle
import readDataMulti
#import os
local_device_protos = device_lib.list_local_devices()

# print([x.name for x in local_device_protos if (x.device_type == 'GPU' or x.device_type=='CPU')])


MAX_SEQUENCE_LENGTH = 160
    # int(sys.argv[1]) # max number of words in a sentence
memory_dim = 400
    # int(sys.argv[2]) #sentence encoding dimension



def word_back(words,word_index):
    nat_sen = ""

    if words == 0 :
        return ""
    for word, i in word_index.items():

            if i == words and i!=0:
                return word

    return "not found"


sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))



embedding_matrix, enc,enc1,enc2,enc3,enc4,encTest,enc1Test,enc2Test,enc3Test,enc4Test,word_index= readDataMulti.create_con(MAX_SEQUENCE_LENGTH)

print("shapes")
print(enc.shape)
print(enc1.shape)
print(enc2.shape)
print(enc3.shape)
print(enc4.shape)
print(encTest.shape)
print(enc1Test.shape)
print(enc2Test.shape)
print(enc3Test.shape)
print(enc4Test.shape)
print("shapes ended")


# print(enc00:4412])
#
# sen = ""
# for w in enc0[4410]:
#     sen = sen + word_back(w,word_index)
# print(sen)
# input("wait")
# print(enc1[4410:4412])
encoder_inputs = [tf.placeholder(tf.int32, shape=(None,),
                                 name="inp%i" % t)
                  for t in range(MAX_SEQUENCE_LENGTH)]
# Placeholders for input, output and dropout
input_x = tf.placeholder(tf.int32, [None, MAX_SEQUENCE_LENGTH], name="input_x")

decoder_inputs = ([tf.zeros_like(encoder_inputs[0], dtype=np.int32, name="GO")]
                  + encoder_inputs[:-1])


labels1 = [tf.placeholder(tf.int32, shape=(None,),
                         name="labels%i" % t)
          for t in range(MAX_SEQUENCE_LENGTH)]
labels2 = [tf.placeholder(tf.int32, shape=(None,),
                         name="labels%i" % t)
          for t in range(MAX_SEQUENCE_LENGTH)]
labels3 = [tf.placeholder(tf.int32, shape=(None,),
                         name="labels%i" % t)
          for t in range(MAX_SEQUENCE_LENGTH)]
labels4 = [tf.placeholder(tf.int32, shape=(None,),
                         name="labels%i" % t)
          for t in range(MAX_SEQUENCE_LENGTH)]

weights = [tf.ones_like(labels_t, dtype=tf.float32)
           for labels_t in labels1]




dtype = tf.float32


num_encoder_symbols = embedding_matrix.shape[0]
num_decoder_symbols = embedding_matrix.shape[0]
embedding_dim = embedding_matrix.shape[1]
embedding_placeholder = tf.placeholder(tf.float32, [num_encoder_symbols, embedding_dim])
# init = tf.constant(0.0, shape=[num_encoder_symbols, embedding_dim])

# with tf.variable_scope("encoder", reuse=None) as scope:
#
#         cell = tf.contrib.rnn.GRUCell(memory_dim)
#
#
#
#
#
#
#         encoder_cell = tf.contrib.rnn.EmbeddingWrapper(
#                                                         cell, embedding_classes=num_encoder_symbols,
#                                                         embedding_size=embedding_dim)
#
#
#         _, encoder_state = tf.contrib.rnn.static_rnn(encoder_cell, encoder_inputs, dtype=dtype) # encoder state is encoded info

with tf.device('/cpu:0'), tf.name_scope("embedding"):

    WW = tf.constant(embedding_matrix, name="1D-CNNW",dtype=tf.float32)

    embedded_chars = tf.nn.embedding_lookup(WW, input_x)
    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

filter_sizes=[3,5,7,10,20,40]
num_filters = 64
pooled_outputs = []
with tf.variable_scope("encoder", reuse=None) as scope:
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            filter_shape = [filter_size, embedding_dim, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(
                embedded_chars_expanded,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Max-pooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, MAX_SEQUENCE_LENGTH - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            print(pooled.get_shape())
            pooled_outputs.append(pooled)

    # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3 )
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
# def loop_function(prev, _):
#     return prev
enc_dim= h_pool_flat.get_shape()

# dec_state_in = tf.placeholder(tf.float32, [None, memory_dim])
encoder_state = h_pool_flat
memory_dim = int(enc_dim[1])
encoder_state = tf.nn.tanh(encoder_state)

with tf.variable_scope("decoder1", reuse=None) as scope:
    # with tf.device('/cpu:0'):

        cell = tf.contrib.rnn.GRUCell(memory_dim)
        cell = tf.contrib.rnn.OutputProjectionWrapper(cell, num_decoder_symbols)

        # W2 = tf.Variable(tf.constant(0.0, shape=[num_encoder_symbols, embedding_dim]),
        #         trainable=False, name="embedding_rnn_decoder/embedding")

        dec_outputs1, dec_memory1 = embedding_rnn_decoder(decoder_inputs,
                                                encoder_state,
                                                cell,
                                                num_decoder_symbols,
                                                embedding_dim,
                                                feed_previous=True)
with tf.variable_scope("decoder2", reuse=None) as scope:
    # with tf.device('/cpu:0'):

        cell = tf.contrib.rnn.GRUCell(memory_dim)
        cell = tf.contrib.rnn.OutputProjectionWrapper(cell, num_decoder_symbols)

        # W2 = tf.Variable(tf.constant(0.0, shape=[num_encoder_symbols, embedding_dim]),
        #         trainable=False, name="embedding_rnn_decoder/embedding")

        dec_outputs2, dec_memory2 = embedding_rnn_decoder(decoder_inputs,
                                                encoder_state,
                                                cell,
                                                num_decoder_symbols,
                                                embedding_dim,
                                                feed_previous=True)
with tf.variable_scope("decoder3", reuse=None) as scope:
    # with tf.device('/cpu:0'):

        cell = tf.contrib.rnn.GRUCell(memory_dim)
        cell = tf.contrib.rnn.OutputProjectionWrapper(cell, num_decoder_symbols)

        # W2 = tf.Variable(tf.constant(0.0, shape=[num_encoder_symbols, embedding_dim]),
        #         trainable=False, name="embedding_rnn_decoder/embedding")

        dec_outputs3, dec_memory3 = embedding_rnn_decoder(decoder_inputs,
                                                encoder_state,
                                                cell,
                                                num_decoder_symbols,
                                                embedding_dim,
                                                feed_previous=True)
with tf.variable_scope("decoder4", reuse=None) as scope:
    # with tf.device('/cpu:0'):

        cell = tf.contrib.rnn.GRUCell(memory_dim)
        cell = tf.contrib.rnn.OutputProjectionWrapper(cell, num_decoder_symbols)

        # W2 = tf.Variable(tf.constant(0.0, shape=[num_encoder_symbols, embedding_dim]),
        #         trainable=False, name="embedding_rnn_decoder/embedding")

        dec_outputs4, dec_memory4 = embedding_rnn_decoder(decoder_inputs,
                                                encoder_state,
                                                cell,
                                                num_decoder_symbols,
                                                embedding_dim,
                                                feed_previous=True)

loss_encoder_mini =    tf.contrib.legacy_seq2seq.sequence_loss(dec_outputs1, labels1, weights, embedding_matrix.shape[0]) \
                + tf.contrib.legacy_seq2seq.sequence_loss(dec_outputs2, labels2, weights, embedding_matrix.shape[0]) \
                + tf.contrib.legacy_seq2seq.sequence_loss(dec_outputs3, labels3, weights, embedding_matrix.shape[0])\
                       + tf.contrib.legacy_seq2seq.sequence_loss(dec_outputs4, labels4, weights, embedding_matrix.shape[0]) \

loss_encoder =    tf.contrib.legacy_seq2seq.sequence_loss(dec_outputs1, labels1, weights, embedding_matrix.shape[0]) \
                + tf.contrib.legacy_seq2seq.sequence_loss(dec_outputs2, labels2, weights, embedding_matrix.shape[0]) \
                + tf.contrib.legacy_seq2seq.sequence_loss(dec_outputs3, labels3, weights, embedding_matrix.shape[0]) \
                 + tf.contrib.legacy_seq2seq.sequence_loss(dec_outputs4, labels4, weights, embedding_matrix.shape[0]) \




train_list = [] # which part of netwrok to train
train_list_restore = []
train_list_names = []
for v in tf.trainable_variables():
    if "embedding:0" not in v.name: #and "decoder4" in v.name:
        train_list.append(v)
    if "embedding:0" not in v.name and "decoder4" not in v.name:
        train_list_restore.append(v)
print("Trainable tensors")
for v in train_list:

        print(v)
print("Restored tensors")
for v in train_list_restore:

        print(v)


optimizer = tf.train.AdamOptimizer(0.0004)
optimizer_mini = tf.train.AdamOptimizer(0.0004)
train_op_mini = optimizer_mini.minimize(loss_encoder_mini, var_list=train_list,aggregation_method=2)

train_op = optimizer.minimize(loss_encoder, var_list=train_list,aggregation_method=2)



# def decodeSentences(X,secondX):
#
#
#     feed_dict = {dec_state_in: X}
#
#     feed_dict.update({encoder_inputs[t]: secondX[t] for t in range(MAX_SEQUENCE_LENGTH)})
#
#     dec_word_out = sess.run(dec_outputs, feed_dict)
#
#     temp = []
#     for i in range(len(dec_word_out)):
#         for j in range(len(dec_word_out[0])):
#             logits_t = dec_word_out[0][j]
#             temp.append((logits_t.argmax(axis=0)))
#
#
#     return words_back(temp, word_index)


def words_back(sen,word_index):
    nat_sen = ""
    for item in sen:
        for word, i in word_index.items():

            if i == item:

                nat_sen = nat_sen +" " +word

    return nat_sen


def train_enc2(X,L1,L2,L3,L4,trainLoss,checkP): # X: input L:label



   # Dimshuffle to seq_len * batch_size
    X1 = X
    XX1 = L1
    XX2 = L2
    XX3 = L3
    XX4 = L4

    X = np.array(X).T
    L1 = np.array(L1).T
    L2 = np.array(L2).T
    L3 = np.array(L3).T
    L4 = np.array(L4).T
    # print(np.array(X).shape)

    # print(np.array(L).shape)

    feed_dict = {encoder_inputs[t]: X[t] for t in range(MAX_SEQUENCE_LENGTH)}


    feed_dict.update({input_x: X1})
    feed_dict.update({labels1[t]: L1[t] for t in range(MAX_SEQUENCE_LENGTH)})
    feed_dict.update({labels2[t]: L2[t] for t in range(MAX_SEQUENCE_LENGTH)})
    feed_dict.update({labels3[t]: L3[t] for t in range(MAX_SEQUENCE_LENGTH)})
    feed_dict.update({labels4[t]: L4[t] for t in range(MAX_SEQUENCE_LENGTH)})




    if trainLoss:


            sess.run(train_op, feed_dict)

    else:

        lg, encOut = sess.run([loss_encoder, encoder_state], feed_dict) # if not train loss then counting the test loss

        # feed_dict = {encoder_inputs[t]: X[t] for t in range(MAX_SEQUENCE_LENGTH)}
        #
        #
        # # with open('encLong.pickle', 'rb') as output:
        #      # encOut = pickle.load(output)
        #
        # # encOut = np.zeros((200,256))
        # # feed_dict.update({encoder_state: encOut})
        # feed_dict.update({labels[t]: L[t] for t in range(MAX_SEQUENCE_LENGTH)})
        #
        # lg2, dec2 = sess.run([loss_encoder, dec_outputs], feed_dict)
        #
        # print(lg)
        # print(lg2)
        #
        # print(np.array(encOut).shape)
        #
        # # with open('encLong.pickle', 'wb') as output:
        # #         pickle.dump(encOut, output)
        #
        # dec = dec2
        # else:

        #
        # all_dia = []
        #
        # for ind in range(np.array(dec).shape[1]):
        #     word_out = np.array(dec)[:, ind, :]
        #     #
        #     # print(np.array(word_out).shape)
        #
        #     # for mm in range(word_out.shape[1]):
        #
        #     temp = []
        #     for nn in range(word_out.shape[0]):
        #             logits_t = word_out[nn, :]
        #
        #             temp.append((np.argmax(logits_t)))
        #
        #
        #     all_dia.append(temp)

        # print(XX[0])
        # print(all_dia[0])
        # counter = 0
        #
        #
        # for j in range(len(XX)):
        #     check = False
        #     ori = ""
        #     rest = ""
        #     for i, ii in zip(XX[j],all_dia[j]):
        #
        #
        #         # print(i, ii)
        #         # ori = ori + word_back(i,word_index)
        #         # rest = rest + word_back(ii,word_index)
        #         if i !=ii:
        #             counter = counter + 1


            #
            # print(ori)
            # print(rest)
            # input("wait")
            # sen = words_back(XX[j], word_index)
            # print(sen)
            # sen = words_back(all_dia[j],word_index)
            # print(sen)
            # input("wait")
        #     if check:
        #         counter = counter + 1
        #
        # sen = words_back(X1[0],word_index)
        # print(sen)
        # sen = words_back(XX[0],word_index)
        # print(sen)
        #
        # sen = words_back(all_dia[0],word_index)
        #
        # print(sen)
        #
        # print(X[:,0])
        # print(counter)

        return lg

def train_enc(X,L1,L2,L3,L4,trainLoss,trainMini):



   # Dimshuffle to seq_len * batch_size
    X1 = X
    XX1 = L1
    XX2 = L2
    XX3 = L3
    XX4 = L4

    X = np.array(X).T
    L1 = np.array(L1).T
    L2 = np.array(L2).T
    L3 = np.array(L3).T
    L4 = np.array(L4).T
    # print(np.array(X).shape)

    # print(np.array(L).shape)

    feed_dict = {encoder_inputs[t]: X[t] for t in range(MAX_SEQUENCE_LENGTH)}


    feed_dict.update({input_x:X1})
    feed_dict.update({labels1[t]: L1[t] for t in range(MAX_SEQUENCE_LENGTH)})
    feed_dict.update({labels2[t]: L2[t] for t in range(MAX_SEQUENCE_LENGTH)})
    feed_dict.update({labels3[t]: L3[t] for t in range(MAX_SEQUENCE_LENGTH)})
    feed_dict.update({labels4[t]: L4[t] for t in range(MAX_SEQUENCE_LENGTH)})

    if trainLoss:

            if trainMini:
                sess.run(train_op_mini, feed_dict)
            else:
                sess.run(train_op, feed_dict)
            return
    else:

        lg,dec1,dec2,dec3,dec4,encOut = sess.run([loss_encoder,dec_outputs1,dec_outputs2,dec_outputs3,dec_outputs4,encoder_state], feed_dict) # encout:encoded info and decoder output

        # feed_dict = {encoder_inputs[t]: X[t] for t in range(MAX_SEQUENCE_LENGTH)}
        #
        #
        # with open('encLong.pickle', 'rb') as output:
        #       encOut = pickle.load(output)
        # #
        # # # encOut = np.zeros((200,256))
        # feed_dict.update({encoder_state: encOut})
        # feed_dict.update({labels[t]: L[t] for t in range(MAX_SEQUENCE_LENGTH)})
        # #
        # lg2, dec2 = sess.run([loss_encoder, dec_outputs], feed_dict)
        #
        # print(lg)
        # print(lg2)
        #
        # print(np.array(encOut).shape)
        #
        # with open('encLong.pickle', 'wb') as output:
        #          pickle.dump(encOut, output)
        #
        # dec = dec2
        # else:

        counter1 = 0
        counter2 = 0
        counter3 = 0
        counter4 = 0
        all_dia1 = []

        for ind in range(np.array(dec1).shape[1]):
            word_out = np.array(dec1)[:, ind, :]
            #
            # print(np.array(word_out).shape)

            # for mm in range(word_out.shape[1]):

            temp = []
            for nn in range(word_out.shape[0]):
                    logits_t = word_out[nn, :]

                    temp.append((np.argmax(logits_t)))


            all_dia1.append(temp)

        # print(XX[0])
        # print(all_dia[0])



        for j in range(len(XX1)):
            check = False
            ori = ""
            rest = ""
            for i, ii in zip(XX1[j],all_dia1[j]):


                # print(i, ii)
                # ori = ori + word_back(i,word_index)
                # rest = rest + word_back(ii,word_index)
                if i !=ii:
                    counter1 = counter1 + 1


            #
            # print(ori)
            # print(rest)
            # input("wait")
            # sen = words_back(XX[j], word_index)
            # print(sen)
            # sen = words_back(all_dia[j],word_index)
            # print(sen)
            # input("wait")
        #     if check:
        #         counter = counter + 1
        #
        # sen = words_back(X1[0],word_index)
        # print(sen)
        # sen = words_back(XX[0],word_index)
        # print(sen)
        #
        # sen = words_back(all_dia[0],word_index)
        #
        # print(sen)
        #
        # print(X[:,0])
        # print(counter)
        all_dia2 = []

        # print(ori)
        # print(rest)
        # input("wait")

        for ind in range(np.array(dec2).shape[1]):
            word_out = np.array(dec2)[:, ind, :]
            #
            # print(np.array(word_out).shape)

            # for mm in range(word_out.shape[1]):

            temp = []
            for nn in range(word_out.shape[0]):
                logits_t = word_out[nn, :]

                temp.append((np.argmax(logits_t)))

            all_dia2.append(temp)

        # print(XX[0])
        # print(all_dia[0])



        for j in range(len(XX2)):
            check = False
            ori = ""
            rest = ""
            for i, ii in zip(XX2[j], all_dia2[j]):

                # print(i, ii)
                # ori = ori + word_back(i,word_index)
                # rest = rest + word_back(ii,word_index)
                if i != ii:
                    counter2 = counter2 + 1

        all_dia3 = []
        # print(ori)
        # print(rest)
        # input("wait")
        for ind in range(np.array(dec3).shape[1]):
            word_out = np.array(dec3)[:, ind, :]
            #
            # print(np.array(word_out).shape)

            # for mm in range(word_out.shape[1]):

            temp = []
            for nn in range(word_out.shape[0]):
                logits_t = word_out[nn, :]

                temp.append((np.argmax(logits_t)))

            all_dia3.append(temp)

        # print(XX[0])
        # print(all_dia[0])



        for j in range(len(XX3)):
            check = False
            ori = ""
            rest = ""
            for i, ii in zip(XX3[j], all_dia3[j]):

                # print(i, ii)
                # ori = ori + word_back(i,word_index)
                # rest = rest + word_back(ii,word_index)
                if i != ii:
                    counter3 = counter3 + 1

        #
        # print(ori)
        # print(rest)
        # input("wait")
        all_dia4 = []

        for ind in range(np.array(dec4).shape[1]):
            word_out = np.array(dec4)[:, ind, :]
            #
            # print(np.array(word_out).shape)

            # for mm in range(word_out.shape[1]):

            temp = []
            for nn in range(word_out.shape[0]):
                logits_t = word_out[nn, :]

                temp.append((np.argmax(logits_t)))

            all_dia4.append(temp)

        # print(XX[0])
        # print(all_dia[0])



        for j in range(len(XX4)):
            check = False
            ori = ""
            rest = ""
            for i, ii in zip(XX4[j], all_dia4[j]):

                # print(i, ii)
                ori = ori + word_back(i,word_index)
                rest = rest + word_back(ii,word_index)
                if i != ii:
                    counter4 = counter4 + 1
        # print(ori)
        # print(rest)
        # input("wait")
        return counter1, counter2, counter3,counter4





def train_sentence_encoder():
    COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'coral', 'black', 'purple', 'pink',
              'brown', 'orange', 'teal', 'yellow', 'lightblue', 'lime', 'lavender', 'turquoise',
              'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']
    min_error = 150000.0
    idx0 = np.arange(enc.shape[0])
    idx1 = np.arange(enc.shape[0])


    total_train_loss = 0
    total_encoder_loss = 0

    tot_train = []
    tot_test1 = []
    tot_test2 = []
    tot_test3 = []
    tot_test4 = []

    tot_test_gold = []
    tot_test_dust = []


    #batch = 256
    # print("training sentence encoder started " + str(datetime.now()))
    minEr = 1000
    for yy in range(0,2000):
        print("training sentence encoder started " + str(datetime.now()))
        np.random.shuffle(idx0)
        if yy%2==0:

            batch = 128
            # print("///////////test error large")
            #
            # sum = 0
            # #
            #
            # for i in range(int(encTest.shape[0] / batch)):
            # # for i in range(100):
            #     sum = sum + train_enc2(encTest[i * batch: (i + 1) * batch, :],
            #                            enc1Test[i * batch: (i + 1) * batch, :],
            #                            enc2Test[i * batch: (i + 1) * batch, :],
            #                            enc3Test[i * batch: (i + 1) * batch, :],
            #                            enc4Test[i * batch: (i + 1) * batch, :],
            #
            #                            False, True)
            #     # int(sys.argv[3]) == 0 :
            #
            # print(sum)

            print("test bit error")
            sum = 0
            sc1 = 0
            sc2 = 0
            sc3 = 0
            sc4 = 0
            for j in range(10):
                    # range(int(encTest.shape[0] / batch))

                c1,c2,c3,c4 = train_enc(encTest[j * batch: (j + 1) * batch, :], enc1Test[j * batch: (j + 1) * batch, :],enc2Test[j * batch: (j + 1) * batch, :],enc3Test[j * batch: (j + 1) * batch, :],enc4Test[j * batch: (j + 1) * batch, :],
                                      False, True)

                sc1 += c1
                sc2 += c2
                sc3 += c3
                sc4 += c4

            tot_ent = batch*10

            sc1 = sc1 // tot_ent
            sc2 = sc2 // tot_ent
            sc3 = sc3 // tot_ent
            sc4 = sc4 // tot_ent
            with open('resultsTest.txt', 'a') as the_file:
                    the_file.write(str(sc1) + '\n')
                    the_file.write(str(sc2) + '\n')
                    the_file.write(str(sc3) + '\n')
                    the_file.write(str(sc4) + '\n')
                    the_file.write('\n')
                # print("///////////training error large")
            #
            # sum = 0
            # for i in range(int(enc.shape[0] / batch)):
            #     sum = sum + train_enc2(enc[i * batch: (i + 1) * batch, :],
            #                            enc1[i * batch: (i + 1) * batch, :], enc2[i * batch: (i + 1) * batch, :],
            #                            enc3[i * batch: (i + 1) * batch, :], enc4[i * batch: (i + 1) * batch, :],
            #                            False, True)
            #     # int(sys.argv[3]) == 0 :
            #
            # print(sum)
            tot_test1.append(sc1)
            tot_test2.append(sc2)
            tot_test3.append(sc3)
            tot_test4.append(sc4)

            if len(tot_test1) > 4:
             fig = plt.figure(figsize=(8, 10))
             plt.plot(tot_test1, label="Doctor View", color=COLORS[0])
             plt.plot(tot_test2, label="Caregiver View", color=COLORS[1])
             plt.plot(tot_test3, label="Family View", color=COLORS[2])
             plt.plot(tot_test4, label="Researcher View", color=COLORS[3])
             plt.xlim(0, len(tot_test1))
             plt.title("Character Recovery Error Per Entry")

             plt.ylabel("Training Epochs")
             plt.tight_layout()
             plt.legend(loc="upper right")
             plt.grid(color="black", which="major", axis="y", linestyle="solid")

             ax = fig.add_subplot(1, 1, 1)

            # Major ticks every 20, minor ticks every 5
             major_ticks = np.arange(0, len(tot_test1), len(tot_test1) / 2)
            # plt.ticklabel_format(style='sci', axis='x', scilimits=(2, 2))

             ax.set_xticks(major_ticks)

             major_ticks = np.arange(0, 160, 160 / 8)

             ax.set_yticks(major_ticks)

            # And a corresponding grid
             ax.grid(True)

             fig.savefig('Test_Error_2.pdf')  # save the figure to file
             plt.close(fig)



        print("test finished")
        saver_all.save(sess,'/home/singhd/PycharmProjects/Privacy/PrivCom-master/') #/Gan512ViewMultiDec4/my-model-512')
        batch = 64
        # print(sum)

        # print("test error large")
        #
        # sum = 0
        # for i in range(int(encTest.shape[0] / batch)):
        #     sum = sum + train_enc2(encTest[i * batch: (i + 1) * batch, :],
        #                            enc1Test[i * batch: (i + 1) * batch, :],enc2Test[i * batch: (i + 1) * batch, :],enc3Test[i * batch: (i + 1) * batch, :],enc4Test[i * batch: (i + 1) * batch, :],
        #                            False, True)
        #     # int(sys.argv[3]) == 0 :
        #
        # print(sum)
        # print("test error")
        # sum = 0
        # for j in range(200):
        #     # int(enc0Test.shape[0] / batch)):
        #     sum = sum + train_enc(enc0Test[j * batch: (j + 1) * batch, :], enc1Test[j * batch: (j + 1) * batch, :],
        #                           False, True)
        #
        # print(sum)

        # print("test error")
        # sum = 0
        # for j in range(int(enc0Test.shape[0] / batch)):
        #     sum = sum + train_enc(enc0Test[j * batch: (j + 1) * batch, :], enc1Test[j * batch: (j + 1) * batch, :],
        #                           False, True)
        #
        # print(sum)

        for j in range(int(enc.shape[0] / batch)):

            if yy < 200:

                train_enc(enc[idx0[j * batch : (j+1) * batch], :],enc1[idx0[j * batch : (j+1) * batch], :],
                          enc2[idx0[j * batch : (j+1) * batch], :],enc3[idx0[j * batch : (j+1) * batch], :],enc4[idx0[j * batch : (j+1) * batch], :], True,True)
            # else:
            #     train_enc(enc[idx0[j * batch: (j + 1) * batch], :], enc1[idx0[j * batch: (j + 1) * batch], :],
            #               enc2[idx0[j * batch: (j + 1) * batch], :], enc3[idx0[j * batch: (j + 1) * batch], :],
            #               enc4[idx0[j * batch: (j + 1) * batch], :], True, False)
            # # if j %1000 ==0:
            #     print("training error small")
            #     np.random.shuffle(idx1)
            #     sum = 0
            #     for i in range(10):
            #         sum = sum + train_enc2(enc[idx1[i * batch: (i + 1) * batch], :], enc1[idx1[i * batch: (i + 1) * batch], :],enc2[idx1[i * batch: (i + 1) * batch], :],enc3[idx1[i * batch: (i + 1) * batch], :],enc4[idx1[i * batch: (i + 1) * batch], :],
            #                               False, True)
            #
            #     #     print("test error")
            # #     sum = 0
            # #     for j in range(200):
            # #         # int(enc0Test.shape[0] / batch)):
            # #         sum = sum + train_enc(enc0Test[j * batch: (j + 1) * batch, :], enc1Test[j * batch: (j + 1) * batch, :],
            # #                               False, True)
            #
            #     print(sum)










    return



#batch = 256
sess.run(tf.global_variables_initializer())
#saver = tf.train.Saver(var_list=train_list_restore)
saver = tf.train.Saver(var_list=train_list)
#saver2 = tf.train.Saver(var_list=train_list)
saver_all = tf.train.Saver()
#saver.restore(sess,tf.train.latest_checkpoint('/home/singhd/PycharmProjects/Privacy/PrivCom-master/'))

#saver2.restore(sess,tf.train.latest_checkpoint('Models/Gan512ViewMultiDec4/'))




# var_1 = [v for v in tf.global_variables() if v.name == "encoder/rnn/embedding_wrapper/embedding:0"][0]
var_2 = [v for v in tf.global_variables() if v.name == "decoder1/embedding_rnn_decoder/embedding:0"][0]
var_3 = [v for v in tf.global_variables() if v.name == "decoder2/embedding_rnn_decoder/embedding:0"][0]
var_4 = [v for v in tf.global_variables() if v.name == "decoder3/embedding_rnn_decoder/embedding:0"][0]
var_5 = [v for v in tf.global_variables() if v.name == "decoder4/embedding_rnn_decoder/embedding:0"][0]
# e1 = var_1.eval(session=sess)
e2 = var_2.eval(session=sess)
e3 = var_3.eval(session=sess)
e4 = var_4.eval(session=sess)
e5 = var_5.eval(session=sess)
# print("before emb 1 = " , e1[0:10,0:10])
# print("before emb 2 = " , e2[0:10,0:10])
# print(" embedding_mat  = " , embedding_matrix[0:10,0:10])
#
#
#
#
# op1 =  var_1.assign(embedding_matrix)
op2 = var_2.assign(embedding_matrix)
op3 = var_3.assign(embedding_matrix)
op4 = var_4.assign(embedding_matrix)
op5 = var_5.assign(embedding_matrix)
# sess.run(op1)
sess.run(op2)
sess.run(op3)
sess.run(op4)
sess.run(op5)
# e1 = var_1.eval(session=sess)
e2 = var_2.eval(session=sess)
e3 = var_3.eval(session=sess)
e4 = var_4.eval(session=sess)
e5 = var_5.eval(session=sess)
#
# print(" after emb 1 = " , e1[0:10,0:10])
# print(" after emb 2 = " , e2[0:10,0:10])
#



if not os.path.exists('/home/singhd/PycharmProjects/Privacy/PrivCom-master/'):
    os.makedirs('/home/singhd/PycharmProjects/Privacy/PrivCom-master/')



nb_epoch = 2

for ep in range(nb_epoch):

    train_sentence_encoder()

