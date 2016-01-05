import os
import io
import json
from PIL import Image
import tornado.ioloop
import tornado.web
import tornado.websocket
from hashlib import sha1
import random
import datetime
import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import pickle
import copy
import cupy

#turn on gpu
try:
  cuda.check_cuda_available()
  cuda.get_device(0).use()
  xp = cupy
except:
  xp = np

simulator_clients = []
view_clients = []
REMOTE_MODE = "TEXTURE" # TEXTURE or DISTANCE
DISTANCE_INPUT = 181
DISTANCE_MAX = 400

MODE = "MANUAL"
ALPHA = 0.001
GAMMA = 0.7
EPSILON = 1.0
NUM_OF_ACTIONS = 4
HISTORY_FILE = "history.pickle"
MODEL_FILE = "DQN.model"

##################################
# Convolutional Neural Network
##################################
class CNN(Chain):
  def __init__(self):
    super(CNN, self).__init__(
      l1=L.Convolution2D(3, 16, ksize=8, stride=4, wscale=np.sqrt(2)),
      l2=L.BatchNormalization(16),
      l3=L.Convolution2D(16, 32, ksize=4, stride=2, wscale=np.sqrt(2)),
      l4=L.BatchNormalization(32),
      l5=L.Linear(4480, 256),
      l6=L.Linear(256, NUM_OF_ACTIONS, initialW=np.zeros((NUM_OF_ACTIONS, 256), dtype=np.float32))
    )

  def __call__(self, x, train):
    h1 = self.l1(x)
    h2 = F.relu(self.l2(h1, test=not train))
    h3 = self.l3(h1)
#    h4 = F.dropout(F.max_pooling_2d(F.relu(self.l4(h3, test=not train)), 2), ratio=0.25, train=train)
    h4 = F.relu(self.l4(h3, test=not train))
    h5 = self.l5(h4)
    y = self.l6(h5)

    return y

##################################
# Multi Layer Perceptron
##################################
class MLP(Chain):
  def __init__(self):
    super(MLP, self).__init__(
      l1=L.Linear(181, 724),
      l2=L.Linear(724, 724),
      l3=L.Linear(724, NUM_OF_ACTIONS)
    )

  def __call__(self, x, train):
    h1 = F.relu(self.l1(x))
    h2 = F.relu(self.l2(h1))
    y = F.relu(self.l3(h2))

    return y



"""
class CNN(Chain):
  def __init__(self):
    super(CNN, self).__init__(
      l1=L.Convolution2D(4, 64, 3, pad=1),
      l2=L.BatchNormalization(64),
      l3=L.Convolution2D(64, 64, 3, pad=1),
      l4=L.BatchNormalization(64),
      l5=L.Convolution2D(64, 128, 3, pad=1),
      l6=L.BatchNormalization(128),
      l7=L.Convolution2D(128, 128, 3, pad=1),
      l8=L.BatchNormalization(128),
      l9=L.Convolution2D(128, 256, 3, pad=1),
      l10=L.BatchNormalization(256),
      l11=L.Convolution2D(256, 256, 3, pad=1),
      l12=L.BatchNormalization(256),
      l13=L.Convolution2D(256, 256, 3, pad=1),
      l14=L.BatchNormalization(256),
      l15=L.Convolution2D(256, 256, 3, pad=1),
      l16=L.BatchNormalization(256),
      l17=L.Linear(4096, 4096),
      l18=L.Linear(4096, 4096),
      l19=L.Linear(49152, 4)
    )

  def __call__(self, x, train):
    print x.data.shape
    print x.data.__class__
    h1 = self.l1(x)
    h2 = F.relu(self.l2(h1, test=not train))
    h3 = self.l3(h1)
    h4 = F.dropout(F.max_pooling_2d(F.relu(self.l4(h3, test=not train)), 2), ratio=0.25, train=train)
    h5 = self.l5(h4)
    h6 = F.relu(self.l6(h5, test=not train))
    h7 = self.l7(h6)
    h8 = F.dropout(F.max_pooling_2d(F.relu(self.l8(h7, test=not train)), 2), ratio=0.25, train=train)
    h9 = self.l9(h8)
    h10 = F.relu(self.l10(h9, test=not train))
    h11 = self.l11(h10)
    h12 = F.relu(self.l12(h11, test=not train))
    h13 = self.l13(h12)
    h14 = F.relu(self.l14(h13, test=not train))
    h15 = self.l15(h14)
    h16 = F.dropout(F.max_pooling_2d(F.relu(self.l16(h15, test=not train)), 2), ratio=0.25, train=train)
    #h17 = F.dropout(F.relu(self.l17(h16)), ratio=0.5, train=train)
    #h18 = F.dropout(F.relu(self.l18(h17)), ratio=0.5, train=train)
    #y = self.l19(h18)
    y = self.l19(h16);
    return y
"""

##################################
# DQN
##################################
class DQN(Chain):
  def __init__(self, predictor):
    super(DQN, self).__init__(predictor=predictor)
    self.Q_func = predictor

  def __call__(self, last_state, last_action, state, reward):
    global ALPHA;
    global GAMMA;

    num_of_batch = state.shape[0]
    _state = Variable(xp.array(state, xp.float32)) # (32, 4, height, width) Variable cupy float32
    q_vector = self.predictor(_state, train=True)  # (32, 4) Variable cupy float32
    _last_state = Variable(xp.array(last_state, xp.float32)) # (32, 4, height, width) Variable cupy float32
    q_vector_last = self.predictor(_last_state, train=True)  # (32, 4) Variable cupy float32

    max_q_value = np.asanyarray(list(map(np.max, q_vector.data.get())), dtype=np.float32) # (32, ) numpy float array
    target_q_data = np.asanyarray(q_vector_last.data.get(), dtype=np.float32) # (32, 4) numpy array

    for i in range(num_of_batch):
      # reward -> (32,) numpy float array
      target_value = reward[i] + GAMMA * max_q_value[i]
      target_q_data[i][last_action] = target_value

    target_q_vector = Variable(xp.array(target_q_data, dtype=xp.float32)) # (32, 4) Variable cupy array
    loss = F.mean_squared_error(target_q_vector, q_vector)
    return loss

  def e_greedy(self, state):
    global NUM_OF_ACTIONS;
    global EPSILON

    q_vector = self.Q_func(state, train=False)

    if(np.random.rand() < EPSILON):
      print "RANDOM ACTION"
      action = np.random.randint(0, NUM_OF_ACTIONS)
    else:
      print "GREEDY ACTION"
      action = np.argmax(q_vector.data)

    return action

##################################
# Experience history
##################################
class History(object):
  def __init__(self, historySize, observeLength, imageHeight, imageWidth):
    global REMOTE_MODE
    global DISTANCE_INPUT

    self.turn = 0
    self.data_index = 0
    self.historySize = historySize
    self.last_action = np.zeros(historySize, dtype=np.int8)
    self.reward = np.zeros(historySize, dtype=np.float32)

    if(REMOTE_MODE == "TEXTURE"):
      self.state = np.zeros((historySize, observeLength, imageHeight, imageWidth), dtype=np.float32)
      self.last_state = np.zeros((historySize, observeLength, imageHeight, imageWidth), dtype=np.float32)
    elif(REMOTE_MODE == "DISTANCE"):
      self.state = np.zeros((historySize, DISTANCE_INPUT), dtype=np.float32)
      self.last_state = np.zeros((historySize, DISTANCE_INPUT), dtype=np.float32)


  # function to stock experience to history buffer
  def stockExperience(self, turn, state, last_state, reward, last_action):
    data_index = turn % self.historySize
    self.state[data_index] = state
    self.last_state[data_index] = last_state
    self.reward[data_index] = reward
    self.last_action[data_index] = last_action
    self.turn = turn
    self.data_index = data_index
    print("experience " + str(data_index) + " has stocked.")

  # getter
  def getNewestState(self):
    return self.state[self.data_index]

  def getNewestLastState(self):
    return self.last_state[self.data_index]



##################################
# Batch
##################################
class Batch(object):
  def __init__(self, state, last_state, last_action, reward):
    self.state = state
    self.last_state = last_state
    self.last_action = last_action
    self.reward = reward

##################################
# WebSocket for DQN simulator
##################################
class DQNWebSocketHandler(tornado.websocket.WebSocketHandler):
  turn = 0  
  saveStep = 1000
  imageWidth = 128
  imageHeight = 96
  observeLength = 3
  historySize = 10**5
  batchSize = 32

  #socket open
  def open(self):
    global REMOTE_MODE

    # load model file
    if(REMOTE_MODE == "TEXTURE"):
      self.model = CNN()
    elif(REMOTE_MODE == "DISTANCE"):
      self.model = MLP()
    self.brain = DQN(self.model)
    try:
      serializers.load_hdf5(MODEL_FILE, self.brain)
      #serializers.save_hdf5("MLP.model", self.brain.predictor)
      print("succeed to load model file")
    except:
      print("failed to load model file")

    # load history file
    try:
      fp = open(HISTORY_FILE, "r")
      self.history = pickle.load(fp)
      self.turn = self.history.turn + 1
      self.state = self.history.getNewestState().copy()
      self.last_state = self.history.getNewestLastState().copy()
      fp.close()
      print "succeed to load history file. [restart from turn " + str(self.turn) +  "]"
    except:
      self.history = History(self.historySize, self.observeLength, self.imageHeight, self.imageWidth)
      self.turn = 0

      if(REMOTE_MODE == "TEXTURE"):
        self.state = np.zeros((self.observeLength, self.imageHeight, self.imageWidth), dtype=np.float32)
        self.last_state = np.zeros((self.observeLength, self.imageHeight, self.imageWidth), dtype=np.float32)
      elif(REMOTE_MODE == "DISTANCE"):
        self.state = np.zeros((DISTANCE_INPUT), dtype=np.float32)
        self.last_state = np.zeros((DISTANCE_INPUT), dtype=np.float32)

      print "failed to load history file"


    self.brain.to_gpu()
    #self.optimizer = optimizers.Adam()
    self.optimizer = optimizers.RMSpropGraves(lr=0.0002, alpha=0.3, momentum=0.2)
    self.optimizer.setup(self.brain)

    # hold connection
    if(self not in simulator_clients):
      simulator_clients.append(self)

  # socket on message
  def on_message(self, data):
    global REMOTE_MODE

    if(REMOTE_MODE == "TEXTURE"):
      self.process_texture(data)
    elif(REMOTE_MODE == "DISTANCE"):
      self.process_distance(data)

    self.turn += 1


  # function to process distance learning
  def process_distance(self, data):
    global NUM_OF_ACTIONS
    global DISTANCE_MAX
    global EPSILON
    global MODE

    print "Turn " + str(self.turn) + " :"

    # send to view
    for client in view_clients:
      client.write_message(data)

    distance_array = json.loads(data)

    # last state( np.array(181, float32) )
    self.last_state = self.state.copy()

    # current state( np.array(181, float32) )
    self.state = np.array(distance_array["distance"], dtype=np.float32) / DISTANCE_MAX

    # last action
    last_action = distance_array["action"]
    print "last_action: " + str(last_action)

    # reward
    reward = distance_array["rewards"]
    print "reward: " + str(reward)

    # store experience
    self.history.stockExperience(self.turn, self.state, self.last_state, reward, last_action)

    # learning from history
    if(MODE == "LEARNING"):
      self.experienceReplay()
    else:
      print "do not learn, only move."

    # determine action from brain on auto run mode
    if(MODE == "LEARNING" or MODE == "FREERUN"):
      if(self.turn <= self.observeLength):
        action = np.random.randint(0, NUM_OF_ACTIONS)
      else:
        _state = Variable(xp.array([self.state], xp.float32))
        action = self.brain.e_greedy(_state)

      self.write_message(str(action))

    # save model and history file
    if(self.turn % self.saveStep == 0):
      print "try to save."
      self.saveData()



  # function to process texture learning
  def process_texture(self, data):
    global NUM_OF_ACTIONS
    global EPSILON
    global MODE

    print "Turn " + str(self.turn) + " :"

    # send to view
    for client in view_clients:
      client.write_message(data, binary=True)

    # stock experience and learn
    imageLength = self.imageWidth * self.imageHeight * 4;
    inputData = map(ord, data)

    # last state( np.array(4, height, width, float32) )
    self.last_state = self.state.copy()

    # current state( np.array(4, height, width, float32) )
    self.state = self.reshapeToRGBImage(map(float, inputData[imageLength:imageLength*2]))
    #current_image = self.convertImage(map(float, inputData[imageLength:imageLength*2]))
    #self.state = np.asanyarray([current_image])
    #self.state = np.asanyarray([self.state[1], self.state[2], self.state[3], current_image])

    # last action( int )
    last_action = inputData[imageLength*2]
    print "last_action: " + str(last_action)

    # reward( float )
    reward = inputData[imageLength*2+1] / 255.0;
    if reward < 0.3:
      reward -= 1.0
    print "reward: " + str(reward)

    # store experience
    self.history.stockExperience(self.turn, self.state, self.last_state, reward, last_action)

    # learning from history
    if(MODE == "LEARNING"):
      self.experienceReplay()
    else:
      print "do not learn, only move."

    # determine action from brain on auto run mode
    if(MODE == "LEARNING" or MODE == "FREERUN"):
      if(self.turn <= self.observeLength):
        action = np.random.randint(0, NUM_OF_ACTIONS)
      else:
        _state = Variable(xp.array([self.state], xp.float32))
        action = self.brain.e_greedy(_state)

      self.write_message(str(action))

    # save model and history file
    if(self.turn % self.saveStep == 0):
      print "try to save."
      self.saveData()


  # function to save file
  def saveData(self):
    try:
      # save model file
      serializers.save_hdf5(MODEL_FILE, self.brain)
      print "succeed to save model"

      # save history file
      #fp = open(HISTORY_FILE, "w")
      #self.brain.state.dump(HISTORY_FILE)
      #pickle.dump(self.brain, fp)
      #fp.close()
      #print "succeed to save history."
    except:
      print "failed to save history."

  # function to update weight
  def experienceReplay(self):
    # choose random index from existing history to make a batch
    if(self.turn < self.historySize):
      replay_indexes = np.random.randint(0, self.turn+1, self.batchSize) #[self.turn]
    else:
      replay_indexes = np.random.randint(0, self.historySize, self.batchSize)

    # create a batch
    batch = Batch(
      state=self.history.state[replay_indexes].copy(),
      last_state=self.history.last_state[replay_indexes].copy(),
      last_action=self.history.last_action[replay_indexes].copy(),
      reward=self.history.reward[replay_indexes].copy()
    )

    # loss backward
    self.optimizer.update(self.brain, batch.last_state, batch.last_action, batch.state, batch.reward)
    print "batch " + str(replay_indexes) + " are used for learning."


  # convert image to gray scale
  def grayScale(self, width, height, inputData):
    length = width * height * 4
    outputData = copy.copy(inputData[0:length])

    for i in range(0, length, 4):
      grayscale = inputData[i] * 0.3 + inputData[i + 1] * 0.59 + inputData[i + 2] * 0.11;
      outputData[i] = grayscale;
      outputData[i+1] = grayscale;
      outputData[i+2] = grayscale;
      outputData[i+3] = inputData[i + 3]

    return outputData

  # convert image to edge
  def toEdge(self, width, height, inputData):
    length = width * height * 4;
    outputData = copy.copy(inputData[0:length])
    w = width;
    h = height;


    for y in range(1, h-1):
      for x in range(1, w-1):
        for c in range(0, 3):
          i = (y*w + x)*4 + c;
          outputData[i] = 127 + -inputData[i - w*4 - 4] - inputData[i - w*4] - inputData[i - w*4 + 4] + -inputData[i - 4] + 8*inputData[i] - inputData[i + 4] + -inputData[i + w*4 - 4] -   inputData[i + w*4] - inputData[i + w*4 + 4];
          if(outputData[i] < 0):
            outputData[i] = 0
          elif(outputData[i] > 255):
            outputData[i] = 255

        outputData[(y*w + x)*4 + 3] = 255

    return outputData

  def convertImage(self, image):
    image = self.grayScale(self.imageWidth, self.imageHeight, image) #gray scale
    image = self.toEdge(self.imageWidth, self.imageHeight, image) #edge
    image = np.array(image, dtype=np.float32)[range(0, len(image), 4)] #49152 -> 12288 units
    image /= 255
    image = image.reshape(self.imageHeight, self.imageWidth)
    return image

  # reshape to 3 channel rgb image
  def reshapeToRGBImage(self, image):
    r = np.array(image, dtype=np.float32)[range(0, len(image), 4)].reshape(self.imageHeight, self.imageWidth) / 255
    g = np.array(image, dtype=np.float32)[range(1, len(image), 4)].reshape(self.imageHeight, self.imageWidth) / 255
    b = np.array(image, dtype=np.float32)[range(2, len(image), 4)].reshape(self.imageHeight, self.imageWidth) / 255
    image = np.asanyarray([r, g, b])
    return image



  def on_close(self):
    if self in simulator_clients:
      simulator_clients.remove(self)


##################################
# WebSocket for view client
##################################
class ViewWebSocketHandler(tornado.websocket.WebSocketHandler):
  def open(self):
    if(self not in view_clients):
      view_clients.append(self)

  def on_message(self, data):
    if(MODE == "MANUAL"):
      for simulator in simulator_clients:
        simulator.write_message(data)

  def on_close(self):
    if self in view_clients:
      view_clients.remove(self)


##################################
#WebSocket for setting
##################################
class SettingWebSocketHandler(tornado.websocket.WebSocketHandler):
  def on_message(self, data):
    parameters = json.loads(data);

    # set epsilon and alpha and gamma
    global EPSILON
    global ALPHA
    global GAMMA
    EPSILON = float(parameters["epsilon"])
    ALPHA = float(parameters["alpha"])
    GAMMA = float(parameters["gamma"])

##################################
# WebSocket for command
##################################
class CommandWebSocketHandler(tornado.websocket.WebSocketHandler):
  def on_message(self, data):
    global MODE

    if(MODE == "MANUAL"):
      for simulator in simulator_clients:
        simulator.write_message(str(random.randint(0, 3)))

      MODE = "LEARNING";
    elif(MODE == "LEARNING"):
      MODE = "FREERUN"
    elif(MODE == "FREERUN"):
      MODE = "MANUAL"

    self.write_message(MODE);

##################################
# Main Handler
##################################
class MainHandler(tornado.web.RequestHandler):
  def get(self):
    self.render("index.html")

##################################
#View Handler
##################################
class ViewHandler(tornado.web.RequestHandler):
  def get(self):
    self.render("view.html")



##################################
#Main Function
##################################
settings = {
  "static_path": os.path.join(os.path.dirname(__file__), "static/"),
}

#listen process
application = tornado.web.Application(
  handlers = [
    (r"/", MainHandler),
    (r"/view", ViewHandler),
    (r"/dqn_websocket", DQNWebSocketHandler),
    (r"/view_websocket", ViewWebSocketHandler),
    (r"/setting_websocket", SettingWebSocketHandler),
    (r"/command_websocket", CommandWebSocketHandler),
  ],
  **settings
)

application.listen(8666)
print ("Port opened.")
tornado.ioloop.IOLoop.instance().start()

