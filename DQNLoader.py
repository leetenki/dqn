import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L




##################################
# DQN Model
##################################
class MLP(Chain):
  NUM_OF_ACTIONS = 4
  MAX_DISTANCE = 400
  EPSILON = 0.05

  def __init__(self):
    super(MLP, self).__init__(
      l1=L.Linear(181, 724),
      l2=L.Linear(724, 724),
      l3=L.Linear(724, self.NUM_OF_ACTIONS)
    )

  def __call__(self, x, train):
    h1 = F.relu(self.l1(x))
    h2 = F.relu(self.l2(h1))
    return F.relu(self.l3(h2))

  def e_greedy(self, state):
    _state = Variable(np.array([state], dtype=np.float32) / self.MAX_DISTANCE)
    q_vector = self(_state, train=False)

    if(np.random.rand() < self.EPSILON):
      action = np.random.randint(0, self.NUM_OF_ACTIONS)
    else:
      action = np.argmax(q_vector.data)
    return action
 



##################################
# main
##################################
dqn = MLP()
serializers.load_hdf5("MLP.model", dqn)

# int[181] (0 ~ 400)
state = [193, 197, 201, 206, 210, 215, 220, 226, 232, 238, 238, 240, 247, 255, 263, 272, 281, 291, 292, 299, 312, 325, 339, 363, 364, 363, 363, 363, 362, 362, 362, 361, 361, 360, 360, 359, 359, 358, 358, 357, 356, 356, 355, 354, 354, 353, 354, 354, 355, 356, 356, 357, 358, 358, 359, 359, 360, 360, 361, 361, 362, 362, 362, 363, 363, 363, 364, 364, 364, 364, 365, 365, 365, 365, 365, 365, 366, 366, 310, 308, 307, 305, 258, 222, 223, 223, 224, 225, 225, 198, 193, 189, 185, 181, 177, 174, 171, 168, 164, 162, 159, 156, 154, 152, 149, 147, 145, 143, 141, 139, 137, 135, 134, 132, 130, 129, 127, 126, 125, 123, 122, 121, 120, 120, 119, 118, 117, 116, 114, 113, 112, 111, 110, 110, 109, 108, 108, 109, 110, 110, 111, 112, 113, 114, 114, 115, 116, 186, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 122, 122, 123, 123, 124, 125, 125, 126, 127, 128, 129, 129, 130, 158, 135, 136, 136, 136, 137, 137, 137, 138, 139]
action = dqn.e_greedy(state) # (0: FORWARD, 1: BACK, 2: LEFT, 3: RIGHT)
print action