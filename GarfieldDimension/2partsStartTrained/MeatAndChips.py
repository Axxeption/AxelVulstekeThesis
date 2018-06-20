import re
import PIL
import gym
import numpy as np
import tesserocr
import tflearn
import gensim
from PIL import Image
from tflearn.layers.estimator import regression
from statistics import median, mean
import cv2
import imutils
import universe
import matplotlib.pyplot as plt
import tensorflow as tf
from random import randint
from datetime import datetime

#button object
class Button:
    x = 0
    y = 0
    vector = []

    def __init__(self, x, y, vector):
        self.x = x
        self.y = y
        self.vector = vector

    def __repr__(self):
        string =  self.x , ", " , self.y
        return string


#Hier beginnen we !
print("Reading in...")
modelNLP = None
modelNLP = gensim.models.KeyedVectors.load_word2vec_format('/home/axel/Documents/GoogleNews-vectors-negative300.bin', binary=True, limit=1000000)
print("done with reading")


def preprocess(observation):
    if(observation[0] != None):
        buttonList = []
        observation = np.array(observation[0]['vision'])  # convert list to 3D-array
        img = observation[125:387, 9:270]  # convert to 210-50x160 input (geel ook al uitgefilterd) anders x = 75
        cv2.imwrite('SequenceBefore.png', img)
        image = Image.open('SequenceBefore.png')
        # convert to grayscale
        img = image.convert('L')
        img_np = np.array(img)
        img_np = (img_np > 100) * 255
        img = PIL.Image.fromarray(img_np.astype(np.uint8))
        img = img.resize((int(img.size[0] * 3.5), int(img.size[1] * 3.5)), PIL.Image.BILINEAR)

        with tesserocr.PyTessBaseAPI() as api:
            api.SetImage(img)
            boxes = api.GetComponentImages(tesserocr.RIL.TEXTLINE, True)
            # print('Found {} textline image components.'.format(len(boxes)))
            if(len(boxes) == 5):
                for i, (im, box, _, _) in enumerate(boxes):
                    # im is a PIL image object
                    # box is a dict with x, y, w and h keys
                    api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
                    ocrResult = api.GetUTF8Text().lower()
                    conf = api.MeanTextConf()
                    ocrResult = re.sub('[^a-zA-Z]', '', ocrResult)
                    print("the word is: " , ocrResult)
                    #TODO get correct vector
                    try:
                        # vector = np.random.random(300)
                        vector = modelNLP.wv[ocrResult]
                        # print("Found vector")
                        x__ = box['x'] / 3.3 + 50
                        y__ = box['y'] / 3.3 + 140
                        # print("x: " , x__)
                        # print("y: " , y__)
                        buttonList.append(Button(x__, y__, vector))
                    except KeyError as err:
                        print("Word not found in google...")
                        # return None
        if len(buttonList) == 5:
            return buttonList
        else:
            return None
    return None

def getstate(buttonList):
    vectorList = []
    # dit geeft me een array van allemaal arrays met 1 element in
    # batch, vector, channel =1
    for vec in range(len(buttonList)):
        vectorList.append(buttonList[vec].vector)
    vectorList = np.concatenate(vectorList)
    # print(vectorList)
    vectorList = vectorList.reshape([1, 1500])  # dus 3d vorm, dit moet want een tensor mag niet echt 1d zijn!
    # print(vectorList)
    return vectorList

def plotter(x):
    plt.ylabel('Current average reward')
    plt.xlabel('Number of episodes')
    plt.title('Progress of Garfield task')
    plt.plot(x)
    plt.savefig('ProgressGarfield.png')
#
# env = gym.make('wob.mini.CircleCenter-v0')
env = gym.make('wob.mini.ClickButton-v0')
# env = gym.make('wob.mini.ClickButtonSequence-v0')


env.configure(remotes=1, fps=1,
              vnc_driver='go',
              vnc_kwargs={'encoding': 'tight', 'compress_level': 0,
                          'fine_quality_level': 100, 'subsample_level': 0})
observation = env.reset()
#these are for the RL step
input_size = 300*5 #ik geef 5 buttons in
howfar =0
buttonList = []
# just the scores that met our threshold:
x__ = 100
y__ = 200
totalEpisode = 0
numberSolutionFound =0
# atexit.register(plotter, x, y)
resume = False;
# model = neural_network_model(input_size)
avgReward = 0

tf.reset_default_graph
#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1, input_size],dtype=tf.float32)
# indie vorm verdeeld tussen...
with tf.name_scope("fully_connected"):
    W = tf.Variable(tf.random_uniform([1500,5],0,0.02))
    Qout = tf.matmul(inputs1,W)
    predict = tf.argmax(Qout,1)
    tf.summary.histogram("weigths", W)


# loss = tf.reduce_sum(tf.square(nextQ - Qout))
# loss = tf.losses.softmax_cross_entropy(nextQ, Qout)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,5],dtype=tf.float32)
with tf.name_scope("loss"):
    loss = tf.reduce_sum(tf.square(nextQ - Qout))
    tf.summary.scalar("loss", loss)
with tf.name_scope("train"):
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.0005)
    updateModel = trainer.minimize(loss)

RandomFactor = tf.placeholder(dtype=tf.float32)
tf.summary.scalar("RandomFactor", RandomFactor)

init = tf.global_variables_initializer()
merged_summary = tf.summary.merge_all()
saver = tf.train.Saver()

# Set learning parameters
y = .95
e = 0.3
forgetFactor = .3
num_episode = 0
#create lists to contain total rewards and steps per episode
rewardList = []
rewardListWindow = []
rewardListWindowTotal = []
sumReward = 0
sumRewardWindow = 0
saved = False

with tf.Session() as sess:
    sess.run(init)
    howfar = +1
    buttonList = None
    # just the scores that met our threshold:
    x__ = 100
    y__ = 200
    r = [1]
    r[0] = 0
    writer = tf.summary.FileWriter("/home/axel/Documents/TensorFlowLog/MeatPastaLong/"+ str(datetime.now()))
    writer.add_graph(sess.graph)
    if (saved):
        print("read model in ")
        saver.restore(sess, "selfmadeRL.ckpt")
    else:
        sess.run(init)
    #Just to start!
    while (observation == [None]):
        # env.render()
        # zolang dat er geen observation is kunnen we niets doen dus proberen we dit maar...
        action = [universe.spaces.PointerEvent(x__, y__, 0), universe.spaces.PointerEvent(x__, y__, 1),
                  universe.spaces.PointerEvent(x__, y__, 0), universe.spaces.PointerEvent(200, 300, 0)]
        action = [action for ob in observation]
        observation, reward_n, done_n, info = env.step(action)
        # soms zit hij precies vast hiermee is dit opgelost
        stuck = 0
        print("aan het proberen...")
    if (observation != [None]):
        while(buttonList == None):
            observation, reward_n, done_n, info = env.step(action)
            buttonList = preprocess(observation)  # ik krijg een buttonlist met 5 buttons terug of ofwel none
        state = getstate(buttonList)
    solutionFound = False
    while not solutionFound:
        while not solutionFound:
            # env.render()
            stuck = stuck + 1
            rAll = 0
            while(stuck > 10 or buttonList == None):
                print("I was stuck, but try to solve...")
                x__ = 300
                y__ = 300
                action = [universe.spaces.PointerEvent(x__, y__, 0), universe.spaces.PointerEvent(x__, y__, 1),
                          universe.spaces.PointerEvent(x__, y__, 0), universe.spaces.PointerEvent(200, 300, 0)]
                action = [action for ob in observation]
                observation, reward_n, done_n, info = env.step(action)
                # env.render()
                stuck = 0;
                buttonList = preprocess(observation)  # ik krijg een buttonlist met 5 buttons terug of ofwel none
                # input("Buttonlist was none")
                if(buttonList != None):
                    state = getstate(buttonList)
            stuck = 0;
            d = False
            if(buttonList != None):
                # print("buttonlist to check if not none: " , buttonList)
                action, allQ = sess.run([predict, Qout], feed_dict={inputs1: state})
                #action[0] is heeft de grootste probabilitietieut naar haalt hem gwn uit een array
                action = action[0]
                # heel soms eens random iets doen
                # print("random kans: " , e)
                if np.random.rand(1) < e:
                    print("Random action picked")
                    action = randint(0, 4)
                print("Action on previous state: ", action)
                # input("waiting to do action")
                x__ = buttonList[action].x
                y__ = buttonList[action].y
                xyaction = [universe.spaces.PointerEvent(x__, y__, 0), universe.spaces.PointerEvent(x__, y__, 1),
                          universe.spaces.PointerEvent(x__, y__, 0),  universe.spaces.PointerEvent(200, 300, 0)]
                xyaction = [xyaction for ob in observation]
                observation, r, d, _ = env.step(xyaction)
                print("reward0: " , r , " done: " , d)
                env.render()
                # x__ = 300
                # y__ = 300
                # waitaction = [universe.spaces.PointerEvent(x__, y__, 0), universe.spaces.PointerEvent(x__, y__, 0),
                #               universe.spaces.PointerEvent(x__, y__, 0)]
                # waitaction = [waitaction for ob in observation]
                # observation, r, d, _ = env.step(waitaction)
                # env.render()
                # print("reward1: " , r , " done: " , d)
                buttonList = preprocess(observation)  # ik krijg een buttonlist met 5 buttons terug of ofwel none
                if(buttonList != None):
                    state1 = getstate(buttonList)
                    # Obtain the Q' values by feeding the new state through our network
                    # ge wilt nu alle qvalues voor de volgende terug krijgen zodat je de grootste kan bepalen die heb j enodig voor die formule
                    #hier moet ik die observatie s1 meegeven in de echte dinges!!
                    Q1 = sess.run(Qout, feed_dict={inputs1: state1})

                    # Obtain maxQ' and set our target value for chosen action.
                    #eerst maximum nemen over alle acties
                    maxQ1 = np.max(Q1)
                    targetQ = allQ
                    #Deze iteration update aangepast!
                    targetQ[0, action] = (1 - forgetFactor) * targetQ[0,action] + forgetFactor * ( r[0] + y * maxQ1)
                    # print("Target: " , targetQ)
                    # print("Q from network: ", Q1)
                    # Train our network using target and predicted Q values
                    # hier doe je de backpropagation je moet al die functies bovenaan in elkaar invullen endan zie jdat je deze 2 waarden nodig ehbt (placeholders)
                    # print("before log is the randomfactor: " , e)
                    _, W1 = sess.run([updateModel, W], feed_dict={inputs1: state, nextQ: targetQ, RandomFactor: e})
                    rAll += r[0]
                    state = state1
                # if(action == 3 or action == 4 and not d[0]):
                #     observation, r, d, _ = env.step(xyaction)
                    # print("reward for second time: " , r , " done: " , d)
                if d[0]:
                    print("Episode is done")
                    # print("resetted with: ", r[0])
                    if(r[0] == 0):
                        r[0] = -1  
                    if(r[0] > 0):
                        numberSolutionFound = numberSolutionFound + 1
                        #als 4 maal na elkaar gevonden kunnen we zeggen dat het ok is
                        if(numberSolutionFound>10):
                            solutionFound = True
                    else:
                        numberSolutionFound = 0
                    e = 0.3 / ((num_episode / 200) + 1)
                        #start bij .1 --> hoe lang random..? bij 100 episodes ng amar de helft..!
                        # input("the episode is resetted")
                    env.reset()
                    num_episode += 1
                    sumReward = sumReward + r[0]
                    sumRewardWindow = sumRewardWindow + r[0]
                    rewardList.append(sumReward/num_episode)
                    if num_episode % 3 == 0:
                        [s] = sess.run([merged_summary], feed_dict={inputs1: state, nextQ: targetQ, RandomFactor: e})
                        #Om de 3 keer wegschrijven en tensorboard updaten
                        writer.add_summary(s, num_episode)
                    if(num_episode%10 == 0):
                        gem = sum(rewardList)/num_episode
                        print("This is episode: " , num_episode , "with a current reward of: " , gem  )
                        np.savetxt('currentReward.txt', rewardList, fmt='%.2f', delimiter=',',
                                   header="value no window")
                        save_path = saver.save(sess, "selfmadeRL.ckpt")
                        plotter(rewardList)
                    if(num_episode%50 == 0):
                        tmp = sumRewardWindow/50
                        rewardListWindowTotal.append(tmp)
                        np.savetxt('windowReward.txt', rewardListWindowTotal, fmt='%.2f', delimiter=',',
                                   header="value no window")
                        sumRewardWindow = 0
                    break

