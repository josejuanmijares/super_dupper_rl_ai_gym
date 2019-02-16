import gym
import random
import numpy as np
import tensorflow as tf
from keras import layers
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Model

from collections import deque
from keras.optimizers import RMSprop
from keras import backend as K
from datetime import datetime
import os.path
import time
from keras.models import load_model
from keras.models import clone_model
from keras.callbacks import TensorBoard
import json

FLAGS = tf.app.flags.FLAGS

def fill_up_flags():
    global FLAGS

    with open('config.json','rb') as f:
        temp_dict = json.load(f)
    for key, data in temp_dict.items():

        if data['type'] == 'string':
            tf.app.flags.DEFINE_string(*data['vars'])
