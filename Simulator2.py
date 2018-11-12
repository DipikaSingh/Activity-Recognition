import numpy as np
import codecs
from faker import Faker
import sys
from mlxtend.preprocessing import one_hot
Name = []
Age = np.arange(60,90)
Gender = np.arange(0,2)
Height = np.arange(150,190,5)
Weight = np.arange(60,120,5)
Address = np.arange(4000,5000,1)
PhoneNum = []
Email = []
BSN = []
Occupation = []
Marital = np.arange(0,7)
time =  np.arange(1,100)
bloodP = np.arange(70,190,10)
Glucose = np.arange(3,8)
MedicalHist = []
Wearable = np.arange(1000,7000,200)
PreSensorK = np.arange(0,2)
PreSensorB = np.arange(0,2)
PreSensorBa = np.arange(0,2)
PreSensorL = np.arange(0,2)
Temperature = np.arange(18,26)
Lights = np.arange(0,2)
Window = np.arange(0,2)
Intruder = np.arange(0,2)
Electricity = np.arange(1000,7000,200)
ExternalGate = np.arange(0,2)


# f = open('source.txt', encoding='utf-8', errors='ignore')
# lines = f.readlines()
# name = []
# for xx in lines:
#
#         print(xx)
#         input("wait")

f = codecs.open('name.txt', encoding='utf-8', errors='ignore')
lines = f.readlines()
name = []
for xx in lines:

        name.append(xx.lstrip().rstrip().lower())

f = codecs.open('diseases.txt', encoding='utf-8', errors='ignore')
lines = f.readlines()
dis = []
for xx in lines:

        dis.append(xx.rstrip())



def myround(x, base=5):
    return int(base * round(float(x)/base))


users = []
user = ""
user1 =""
user2 = ""
user3 = ""
user4 = ""
maxl = 0
f0 = open('sourceN.txt', 'w')
f1 = open('view1N.txt', 'w')
f2 = open('view2N.txt', 'w')
f3 = open('view3N.txt', 'w')
f4 = open('view4N.txt', 'w')
f00 = open('sourceNTest.txt', 'w')
f11 = open('view1NTest.txt', 'w')
f22 = open('view2NTest.txt', 'w')
f33 = open('view3NTest.txt', 'w')
f44 = open('view4NTest.txt', 'w')
fake = Faker()
# for nn in range(len(name)):
# for numg in range(int(sys.argv[1])):
for numg in range(10000):
    # print(fake.name())

    gname = fake.name()
    user = user + "n" + ":"+  gname + "|"

    user1 = user1 + "n" + ":" + gname + "|"
    user2 = user2 + "n" + ":"+  gname + "|"
    user3 = user3 + "n" + ":" + gname + "|"
    user4 = user4 + "n" + ":"+  "*" + "|"




    age = np.random.randint(60,90)
    user = user+ "a" + ":"+str(age) + "|"

    user1 = user2 + "a" + ":" + str(age) + "|"
    user2 = user2 + "a" + ":" + str(age) + "|"
    user3 = user3 + "a" + ":" + str(myround(age,10)) + "|"
    user4 = user4 + "a" + ":" + str(myround(age,10)) + "|"






    gen = ["m","f"]
    gender = np.random.randint(0,2)
    user = user+ "g" + ":"+ str(gen[gender]) + "|"

    user1 = user1 + "g" + ":" + str(gen[gender]) + "|"
    user2 = user2 + "g" + ":" + str(gen[gender]) + "|"
    user3 = user3 + "g" + ":" + str(gen[gender]) + "|"
    user4 = user4 + "g" + ":" + str(gen[gender]) + "|"




    height = np.random.randint(150,190)
    user = user+ "h" + ":"+str(height) + "|"

    user1 = user1 + "h" + ":" + str(height) + "|"
    user2 = user2 + "h" + ":" + str(height) + "|"
    user3 = user3 + "h" + ":" + str(myround(height)) + "|"
    user4 = user4 + "h" + ":" + str(myround(height)) + "|"


    weight = np.random.randint(60,120)
    user = user+ "w" + ":"+ str(weight) + "|"

    user1 = user1 + "w" + ":" + str(weight) + "|"
    user2 = user2 + "w" + ":" + str(weight) + "|"
    user3 = user3 + "w" + ":" + str(myround(weight)) + "|"
    user4 = user4 + "w" + ":" + str(myround(weight)) + "|"


    zip = np.random.randint(1000,9900)
    user = user+ "ad" + ":"+str(zip) + "|"

    user1 = user1 + "ad" + ":" + str(zip) + "|"
    user2 = user2 + "ad" + ":" +  str(myround(zip,100)) + "|"
    user3 = user3 + "ad" + ":" + str(zip) + "|"
    user4 = user4 + "ad" + ":" + str(myround(zip,100)) + "|"


    phone = np.random.randint(100000,999999)
    user = user+ "ph" + ":"+str(phone) + "|"

    user1 = user1 + "ph" + ":" + str(phone) + "|"
    user2 = user2 + "ph" + ":" + str(phone) + "|"
    user3 = user3 + "ph" + ":" + str(phone) + "|"
    user4 = user4 + "ph" + ":" + "unk" + "|"


    mar = np.random.randint(0,7)
    user = user+ "m" + ":"+ str(mar) + "|"

    user1 = user1+ "m" + ":"+ str(mar) + "|"
    if mar >0:
        mar = 1
    user2 = user2+ "m" + ":"+ str(mar) + "|"
    user3 = user3+ "m" + ":"+ str(mar) + "|"
    user4 = user4+ "m" + ":"+ str(mar) + "|"



    occ = np.random.randint(0,20)
    user = user+ "oc" + ":"+ str(occ) + "|"

    user1 = user1 + "oc" + ":" + str(occ) + "|"
    user2 = user2 + "oc" + ":" + str(occ%5) + "|"
    user3 = user3 + "oc" + ":" + str(occ%5) + "|"
    user4 = user4 + "oc" + ":" + str(occ%5) + "|"


    nn = np.random.randint(0,len(dis))
    user = user+ "ds" + ":"+  dis[nn] + "|"

    user1 = user1 + "ds" + ":" + dis[nn] + "|"
    user2 = user2 + "ds" + ":" + dis[nn] + "|"
    user3 = user3 + "ds" + ":" + dis[nn] + "|"
    diss = dis[nn]
    if nn >0 and nn<6:
        diss  = "heart"
    if nn >6 and nn<11:
        diss  = "lung"
    if nn >11 and nn<15:
        diss  = "dementia"


    user4 = user4 + "ds" + ":" + diss + "|"


    tuser = user
    tuser1 = user1
    tuser2 = user2
    tuser3 = user3
    tuser4 = user4

    for i in range(100,200):

        user = tuser

        user1 = tuser1
        user2 = tuser2
        user3 = tuser3
        user4 = tuser4

        # ts = np.random.randint(100, 999)
        user = user + "ts" + ":" + str(i) + "|"

        user1 = user1 + "ts" + ":" + str(i) + "|"
        user2 = user2 + "ts" + ":" + str(i) + "|"
        user3 = user3 + "ts" + ":" + str(i) + "|"
        user4 = user4 + "ts" + ":" + str(i) + "|"

        bp = np.random.randint(70, 190)
        user = user + "bp" + ":" + str(bp) + "|"

        user1 = user1 + "bp" + ":" + str(myround(bp,10)) + "|"
        user2 = user2 + "bp" + ":" + str(bp) + "|"
        user3 = user3 + "bp" + ":" + str(bp) + "|"
        user4 = user4 + "bp" + ":" +str(myround(bp,10)) + "|"


        gc = np.random.randint(3, 8)
        user = user + "gc" + ":" + str(gc) + "|"


        if gc <5:
            user1 = user1 + "gc" + ":" + "l" + "|"
            user4 = user4 + "gc" + ":" + "l" + "|"
        if gc > 5:
            user1 = user1 + "gc" + ":" + "h" + "|"
            user4 = user4 + "gc" + ":" + "h" + "|"
        user2 = user2 + "gc" + ":" + str(gc) + "|"
        user3 = user3 + "gc" + ":" + str(gc) + "|"


        we = np.random.randint(1000,7000)
        user = user + "we" + ":" + str(we) + "|"

        user1 = user1 + "we" + ":" + str(we) + "|"
        user2 = user2 + "we" + ":" + str(we) + "|"
        user3 = user3 + "we" + ":" + str(we) + "|"
        user4 = user4 + "we" + ":" + str(we) + "|"

        sen = np.random.randint(1, 5)
        user = user + "ss" + ":" + str(sen) + "|"

        user1 = user1 + "ss" + ":" + str(sen) + "|"
        user2 = user2 + "ss" + ":" + "unk" + "|"
        user3 = user3 + "ss" + ":" + str(sen) + "|"
        user4 = user4 + "ss" + ":" + str(sen) + "|"


        temp = np.random.randint(16, 30)
        user = user + "tp" + ":" + str(temp) + "|"

        user1 = user1 + "tp" + ":" + str(temp) + "|"
        user2 = user2 + "tp" + ":" + str(temp) + "|"
        user3 = user3 + "tp" + ":" + str(temp) + "|"
        user4 = user4 + "tp" + ":" + str(myround(temp)) + "|"

        light = np.random.randint(1, 5)
        user = user + "l" + ":" + str(light) + "|"

        user1 = user1 + "l" + ":" + str(light) + "|"
        user2 = user2 + "l" + ":" + "unk" + "|"
        user3 = user3 + "l" + ":" + "unk" + "|"
        user4 = user4 + "l" + ":" + str(light) + "|"

        window = np.random.randint(1, 3)
        user = user + "win" + ":" + str(window) + "|"

        user1 = user1 + "wn" + ":" + str(window) + "|"
        user2 = user2 + "wn" + ":" + "unk" + "|"
        user3 = user3 + "wn" + ":" + str(window) + "|"
        user4 = user4 + "wn" + ":" + "unk" + "|"


        # i = np.random.randint(0, 2)
        # user = user + "int" + ":" + str(i) + "|"
        # user1 = user1 + "int" + ":" + str(i) + "|"
        # user2 = user2 + "int" + ":" + "unk" + "|"
        # user3 = user3 + "int" + ":" + str(i) + "|"
        # user4 = user4 + "int" + ":" + "unk" + "|"


        el = np.random.randint(1000,7000)
        user = user + "el" + ":" + str(el) + "|"

        user1 = user1 + "el" + ":" + str(myround(el,500)) + "|"
        user2 = user2 + "el" + ":" + "unk" + "|"
        user3 = user3 + "el" + ":" + "unk" + "|"
        user4 = user4 + "el" + ":" + str(myround(el,500)) + "|"




        ext = np.random.randint(0, 2)
        user = user + "ex" + ":" + str(ext) + "|"

        user1 = user1 + "ex" + ":" + str(ext) + "|"
        user2 = user2 + "ex" + ":" + "unk" + "|"
        user3 = user3 + "ex" + ":" + str(ext) + "|"
        user4 = user4 + "ex" + ":" + "unk" + "|"

        user = user + "eos"
        user1 = user1 + "eos"
        user2 = user2 + "eos"
        user3 = user3 + "eos"

        user4 = user4 + "eos"


        if len(user)>maxl:
            maxl = len(user)

        tort = np.random.randint(0, 100)
        if tort<80:
            #print(user ,file = f0)
            f0.write(user + "\n")

            #print(user1 , file=f1)
            f1.write(user1 + "\n")

            #print(user2 , file=f2)
            f2.write(user2 + "\n")


            #print(user3 , file=f3)
            f3.write(user3 + "\n")


            #print(user4, file=f4)
            f4.write(user4 + "\n")

        else:
            #print(user ,file = f00)
            f00.write(user + "\n")


            #print(user1 , file=f11)
            f11.write(user1 + "\n")


            #print(user2 , file=f22)
            f22.write(user2 + "\n")


            #print(user3 , file=f33)
            f33.write(user3 + "\n")


            #print(user4, file=f44)
            f44.write(user4 + "\n")





    user = ""
    user1 = ""
    user2 = ""
    user3 = ""
    user4 = ""
print(maxl)
f0.close()
f1.close()
f2.close()
f3.close()
f4.close()

f00.close()
f11.close()
f22.close()
f33.close()
f44.close()