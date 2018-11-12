import gzip
import codecs
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow.python.client import device_lib

#local_device_protos = device_lib.list_local_devices()

from nltk.tokenize import word_tokenize
#print([x.name for x in local_device_protos if (x.device_type == 'GPU' or x.device_type=='CPU')])

MAX_NB_WORDS = 4000000
MAX_Conv_LENGTH = 10 # max number of sentences in a dialogue
MAX_SEQUENCE_LENGTH = 400 # max number of words in a sentence




def createSentenceEncoding(XX,sen_dic,num):




    X = np.zeros((MAX_SEQUENCE_LENGTH*num),dtype=int)
    counter = 0
    for index_sen in XX:


                if index_sen != 0 and str(index_sen)  in sen_dic:

                    X[counter*MAX_SEQUENCE_LENGTH:(counter+1)*MAX_SEQUENCE_LENGTH] = sen_dic[str(index_sen)]

                    counter = counter + 1
                #     # X.append(np.zeros(MAX_SEQUENCE_LENGTH),axis=1)
                #
                # else:
                #     if str(index_sen) in sen_dic:
                #         # X.append(sen_dic[str(index_sen)],axis=1)



    # X = np.array(X)


    return X

def parse(filename):
        f = gzip.open(filename, 'r')
        entry = {}
        postexts = []
        midtexts = []
        negtexts = []
        counter =  0
        for l in f:

            if True:
                counter = counter +1
                l = l.decode("utf-8").strip()

                colonPosStart = l.find('reviewText')
                colonPosEnd = l.find('", "overall":')

                colonStart = l.find('overall')
                colonEnd = l.find(', "summary":')
                rev = l[colonStart + 10:colonEnd]
                text =l[colonPosStart + 14:colonPosEnd]
                # sent_tokenize_list = sent_tokenize(text)

                if "great illustrations" in text and rev == "1.0":

                    if len(word_tokenize(text)) > 0 and len(word_tokenize(text)) < MAX_SEQUENCE_LENGTH :
                        print(rev)
                        print (l)
                        input("wait")

                # for sen in sent_tokenize_list:
                # if  len(word_tokenize(text)) > 0 and len(word_tokenize(text)) <MAX_SEQUENCE_LENGTH and ( rev == "5.0") :
                #         postexts.append(text + " unk")
                #
                # if  len(word_tokenize(text)) > 0 and len(word_tokenize(text)) <MAX_SEQUENCE_LENGTH and (rev == "1.0") :
                #         negtexts.append(text + " unk")
                #
                # if  len(word_tokenize(text)) > 0 and len(word_tokenize(text)) <MAX_SEQUENCE_LENGTH and (rev == "3.0") :
                #         midtexts.append(text + " unk")

        # with open('pos_text_data.pickle', 'wb') as output:
        #     pickle.dump(postexts, output)
        # with open('mid_text_data.pickle', 'wb') as output:
        #         pickle.dump(midtexts, output)
        # with open('neg_text_data.pickle', 'wb') as output:
        #     pickle.dump(negtexts, output)

        return postexts,midtexts,negtexts
def create_con(MAX_SEQUENCE_LENGTH):


    small = 200000
    tokenizer = Tokenizer(num_words=10000000,lower=True,filters='!"#$%&()+,-./;<=>?@[\\]^_`{}~\t\n',)

    with open('wi_data.pickle', 'rb') as data:
       tokenizer.word_index = pickle.load(data)
    word_index = tokenizer.word_index
    texts = []

    f = codecs.open('sourceN.txt', encoding='utf-8', errors='ignore')
    lines = f.readlines()

    counter = 0
    for xx in lines[:small]:
        texts.append(" ".join(xx))

    f.close()
    f = codecs.open('view1N.txt', encoding='utf-8', errors='ignore')
    lines = f.readlines()

    for xx in lines[:small]:
        texts.append(" ".join(xx))
    f.close()
    f = codecs.open('view2N.txt', encoding='utf-8', errors='ignore')
    lines = f.readlines()

    for xx in lines[:small]:
        texts.append(" ".join(xx))
    f.close()
    f = codecs.open('view3N.txt', encoding='utf-8', errors='ignore')
    lines = f.readlines()

    for xx in lines[:small]:
        texts.append(" ".join(xx))
    f.close()

    f = codecs.open('view4N.txt', encoding='utf-8', errors='ignore')
    lines = f.readlines()

    for xx in lines[:small]:
        texts.append(" ".join(xx))

    f.close()

    small = 20000
    textsTest = []
    f = codecs.open('sourceNTest.txt', encoding='utf-8', errors='ignore')
    lines = f.readlines()

    for xx in lines[:small]:
        textsTest.append(" ".join(xx))
    f.close()
    f = codecs.open('view1NTest.txt', encoding='utf-8', errors='ignore')
    lines = f.readlines()

    for xx in lines[:small]:
        textsTest.append(" ".join(xx))

    f.close()
    f = codecs.open('view2NTest.txt', encoding='utf-8', errors='ignore')
    lines = f.readlines()

    for xx in lines[:small]:
        textsTest.append(" ".join(xx))
    f.close()
    f = codecs.open('view3NTest.txt', encoding='utf-8', errors='ignore')
    lines = f.readlines()

    for xx in lines[:small]:
        textsTest.append(" ".join(xx))
    f.close()
    f = codecs.open('view4NTest.txt', encoding='utf-8', errors='ignore')
    lines = f.readlines()

    for xx in lines[:small]:
        textsTest.append(" ".join(xx))

    f.close()

    print("tokenizer")



    #tokenizer.fit_on_texts(texts[:10000]) #when tokenization done then comment or give first 10000



    #word_index = tokenizer.word_index

    # with open('wi_data.pickle', 'wb') as output:
    #     pickle.dump(word_index, output, protocol=2)


    print("tokenizer done")
    seqall = tokenizer.texts_to_sequences(texts)
    seqallTest = tokenizer.texts_to_sequences(textsTest)
    senall = pad_sequences(seqall, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
    senallTest = pad_sequences(seqallTest, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    print("found words ",len(word_index))
    print(senall[0:1])
    print("shape of senall",np.array(senall).shape)
    print("shape of senall Test", np.array(senallTest).shape)
    print('Found %s unique tokens.' % len(word_index))


    with open('wi_data.pickle', 'rb') as data:
        tokenizer.word_index = pickle.load(data)
    word_index = tokenizer.word_index



    embedding_matrix = np.zeros((len(word_index) + 1, len(word_index) + 1))

    for word, i in word_index.items():

        embedding_vector = np.array([int(j == i) for j in range(len(word_index) + 1)])


        embedding_matrix[i] = embedding_vector






    embedding_matrix[0] = np.zeros(len(word_index) + 1)
    print(len(embedding_matrix))


    with open('senall.pickle', 'wb') as output:
        pickle.dump(senall, output, protocol=2)
    with open('senallTest.pickle', 'wb') as output:
        pickle.dump(senallTest, output, protocol=2)


    with open('senall.pickle', 'rb') as data:
        senall = pickle.load(data)


    with open('senallTest.pickle', 'rb') as data:
        senallTest = pickle.load(data)

    inP = senall[0:int(len(senall)/5)]
    outP1 = senall[int(len(senall)/5):int(len(senall)/5) *2]
    outP2 = senall[int(len(senall) / 5)*2:int(len(senall) / 5)*3]
    outP3 = senall[int(len(senall) / 5)*3:int(len(senall) / 5)*4]
    outP4 = senall[int(len(senall) / 5)*4:]


    inPTest = senallTest[0:int(len(senallTest)/5)]
    outPTest1 = senallTest[int(len(senallTest)/5):int(len(senallTest)/5)*2]
    outPTest2 = senallTest[int(len(senallTest) / 5)*2:int(len(senallTest)/5)*3]
    outPTest3 = senallTest[int(len(senallTest) / 5)*3:int(len(senallTest)/5)*4]
    outPTest4 = senallTest[int(len(senallTest) / 5)*4:]







    # print("after all")
    # print(senall[4410:4412])
    # print(inP[4410:4412])
    # print("after all done")

    # input("wait")

    return embedding_matrix,inP,outP1,outP2,outP3,outP4,inPTest,outPTest1,outPTest2,outPTest3,outPTest4,word_index
