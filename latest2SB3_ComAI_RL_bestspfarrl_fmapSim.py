#from a import *
import multiprocessing
#import gymnasium as gym
import pdb
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
import gymnasium as gym
import time
from gymnasium import spaces
from stable_baselines3 import A2C
#from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO,DQN
import logging
from stable_baselines3.common.env_checker import check_env
from sklearn.utils.linear_assignment_ import linear_assignment
import numpy as np
import random
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
from multiprocessing import Pool
import matplotlib.pyplot as plt4
import matplotlib.pyplot as plt5
import multiprocessing as mp
import sys
import datetime
import argparse
import ast
import json
from PIL import Image
from tqdm import tqdm
import cv2
import itertools
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import os
import concurrent.futures
from stable_baselines3.common.callbacks import BaseCallback
#from stable_baselines3.common.preprocessing import ImagePreprocessor
from YOLOV3.dulangacode.prior_generator import get_prior_batch, get_priors, initialize_priors,calc_accuracy,get_Accuracy_Metrics, get_prior_mask_batch
from ssdKerasMaster.cal_accu_collab_r5_c7_small_large import *
from ssdKerasMaster.misc.ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2
ENABLE_PRINT = False
import time
import glob
image_width,image_height =150,150
import random
import shutil
from collections import OrderedDict
import tensorflow as tf
import threading
np.set_printoptions(threshold=np.inf)
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-20:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Open the text file in read mode
import ast
with open("data_overlap_dic_test.txt", "r") as file:
    # Read all lines from the file
    lines = file.readlines()

# Print the content of the file
main_dic={}
for line in lines:
    #print(line.strip())
    aa= line.strip()
    a_dict = ast.literal_eval(aa)
    keys_as_strings = [str(key) for key in a_dict.keys()]
    name = keys_as_strings[0].split("_")
    main_dic[name[-1]] =a_dict.values()

data_suppix = "_comAI" # if loading train set use "" and if loading test set use "_test"
dataset_load_name="final_sim_images"#"final_sim_images"
dataset_load_name_fmap="final_sim_image_sim_sfmap"

A = [0,10, 20, 30, 40, 50, 60, 70, 80, 90]
B = [0,10, 20, 30, 40, 50, 60, 70, 80, 90]
C = [0,10, 20, 30, 40, 50, 60, 70, 80, 90]

img_width =1311# 1311#1368 #train set 1311   #1280
img_height =651# 651#665 #train set  651

logging.basicConfig(level=logging.INFO, format='%(message)s')
file_handler = logging.FileHandler('./logwithbestPairs.txt')

file_handler.setLevel(logging.INFO)

# Configure the format for the file handler
file_format = logging.Formatter('%(message)s')
file_handler.setFormatter(file_format)


def normalize_image_RL(image):
    # Scale pixel values to the range [0, 1]
    #image = image.astype(np.float32) / 255.0

    # Apply channel-wise normalization
    #mean = np.mean(image, axis=(0, 1))
    #std = np.std(image, axis=(0, 1))
    #image = (image - mean) / std
    resized_image = cv2.resize(image, (150, 150))
    # Convert the resized image to grayscale 
    return resized_image

image_root_path='./YOLOV3/'+dataset_load_name+'/'
image_root_path_fmap='./YOLOV3/'+dataset_load_name_fmap+'/'

cam_images={}
# List all files in the directory
files = os.listdir(image_root_path)
en_load_images=1
def load_input_fmap_RL(path,frameID):
    
    if(len(str(frameID))==1):
     filename = path + "frame_000"+str(frameID)+".npz"
     ww= filename.split("/")[4]+"/"+filename.split("/")[5]
     image = image_root_path + ww+"/frame_000"+str(frameID)+".jpg"
    elif(len(str(frameID))==2):
     filename = path + "frame_00"+str(frameID)+".npz"
     ww= filename.split("/")[4]+"/"+filename.split("/")[5]
     image = image_root_path + ww+"/frame_00"+str(frameID)+".jpg"
    else:
     filename = path + "frame_0"+str(frameID)+".npz"
     ww= filename.split("/")[4]+"/"+filename.split("/")[5]
     image = image_root_path + ww+"/frame_0"+str(frameID)+".jpg"
    #print("loaded image path",image)
    #start_time=time.time() 
    #dd=cv2.imread(image)
    #dd=cv2.resize(dd,(75,75))
    fmap = np.load(filename)
    fmap = 1-((fmap['matrix'])*255)
    #heatmap_normalized = cv2.normalize(np.array(fmap), None, 0, 255, cv2.NORM_MINMAX)
    #heatmap_normalized = np.uint8(heatmap_normalized)
    ## Apply a colormap to the heatmap
    #heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    ## Overlay the heatmap on the image
    #alpha = 0.6  # Transparency factor
    #overlayed_image = cv2.addWeighted(heatmap_colored, alpha, dd, 1 - alpha, 0)
    #overlayed_image=cv2.resize(overlayed_image,(300,300))
    #cv2.imwrite("./iiiii.png", overlayed_image)
    

    #print(fmap.shape)
    return fmap
if (en_load_images==1):
    for a in A:
      print("A"+str(a))
      view1=1 
      foldername_ref  = "A"+str(a)+'/'
      fnal_path_ref = image_root_path_fmap +'/' +foldername_ref + "View_00"+str(view1)+'/'
      for value in range(500):
         index= "A"+str(a)+"_"+str(value)
         input_images_ref = load_input_fmap_RL(fnal_path_ref,value)
         #x = np.arange(0, 75, 1)  # Replace with your x coordinates
         #y = np.arange(0, 75, 1)  # Replace with your y coordinates
         #x, y = np.meshgrid(x, y)
         #print(input_images_ref)
         ##zm =fmap_RL_input.copy() # Replace with your z values
         ### Create a 3D plot
         #fig = plt.figure()
         #ax = fig.add_subplot(111, projection='3d')
         ### Plot the 3D surface
         #ax.plot_surface(x, y, input_images_ref, cmap='viridis')
         ### Add labels
         #ax.set_xlabel('X')
         #ax.set_ylabel('Y')
         #ax.set_zlabel('Z')
         ## Iterate through each element using meshgrid indexing
         #for i in range(75):
         # for j in range(75):
         #   value = input_images_ref[i, j]  # Access value using meshgrid indexing
         #   # You can access y coordinate using Y[i, j] if using the matrix approach
         #   if value > 0:
         #     print(f"Value: {value}, X: {i}, Y: {j}")
         ### Show the plot
         ##fig.canvas.mpl_connect('key_press_event', on_key)
         #plt.savefig('iiiiiiiiii.png')
         #plt.show()
         image_out=normalize_image_RL(input_images_ref)
         #print(image_out)
         cam_images[index]=image_out
      #break
    for a in B:
      view1=2 
      print("B"+str(a))
      foldername_ref  = "B"+str(a)+'/'
      fnal_path_ref = image_root_path_fmap +'/' +foldername_ref + "View_00"+str(view1)+'/'
      for value in range(500):
         index= "B"+str(a)+"_"+str(value)
         input_images_ref = load_input_fmap_RL(fnal_path_ref,value)
         image_out=normalize_image_RL(input_images_ref)
         cam_images[index]=image_out
    for a in C:
      print("C"+str(a))
      view1=3 
      foldername_ref  = "C"+str(a)+'/'
      fnal_path_ref = image_root_path_fmap +'/' +foldername_ref + "View_00"+str(view1)+'/'
      for value in range(500):
         index= "C"+str(a)+"_"+str(value)
         input_images_ref = load_input_fmap_RL(fnal_path_ref,value)
         image_out=normalize_image_RL(input_images_ref)
         cam_images[index]=image_out

# Add the file handler to the logger
logger = logging.getLogger('')
logger.addHandler(file_handler)

## Generate combinations
combinations = []
total_collaboration_tuples=[]
for a in A:
    for b in B:
        for c in C:
            combinations.append((a, b, c))
            total_collaboration_tuples.append(('A'+str(a), 'B'+str(b), 'C'+str(c)))
            total_collaboration_tuples.append(('B'+str(b), 'A'+str(a), 'C'+str(c)))
            total_collaboration_tuples.append(('C'+str(c), 'A'+str(a), 'B'+str(b)))


dic_init_angle_summery = {}
for i, combination in enumerate(combinations):
    dic_init_angle_summery[str(combination)] = 0
tf_dic={} # this dic contains the transfomatons of views
# Set the random seed for reproducibility
random.seed(42)
sampled_combinations=[]



AREA_RANGE = range(0,150*150)
PEOPLE_RANGE = range(0,50)
OCCLUSION_RANGE = np.arange(0, 1.01, 0.01)
start_image_num = 0
final_image_num = 80



MAx_ITER_PER_EPISODE = 489
Summery_action_training = {}
global_test_summery={}
global_test_summery_ABC={}

global_yesSteeringYesComAI_summery ={}
global_NoSteeringNoComAI_summery ={}
global_NoSteeringYesComAI_summery ={}
global_percentageOfcollaboration ={}
global_global_totaldetection_summary={}

global_CAMA_Fscore={}
global_CAMB_Fscore={}
global_CAMC_Fscore={}

#dic_action={0:[1,1,1],1:[0,0,0],2:[0,0,-1],3:[0,0,1],4:[0,-1,0],5:[0,-1,-1],6:[0,-1,1],7:[0,1,0],8:[0,1,-1],9:[0,1,1],10:[-1,0,0],11:[-1,0,-1],12:[-1,0,1],13:[-1,-1,0],14:[-1,-1,-1],15:[-1,-1,1],16:[-1,1,0],17:[-1,1,-1],18:[-1,1,1],19:[1,0,0],20:[1,0,-1],21:[1,0,1],22:[1,-1,0],23:[1,-1,-1],24:[1,-1,1],25:[1,1,0],26:[1,1,-1]}
# update below dictionary with 3 bits of best coollaborators pairs 
dic_action = {0: [1, 1, 1, 1, 1, 1], 1: [1, 1, 1, 1, 1, 0], 2: [1, 1, 1, 1, 0, 1], 3: [1, 1, 1, 1, 0, 0], 4: [1, 1, 1, 0, 1, 1], 5: [1, 1, 1, 0, 1, 0], 6: [1, 1, 1, 0, 0, 1], 7: [1, 1, 1, 0, 0, 0], 8: [0, 0, 0, 1, 1, 1], 9: [0, 0, 0, 1, 1, 0], 10: [0, 0, 0, 1, 0, 1], 11: [0, 0, 0, 1, 0, 0], 12: [0, 0, 0, 0, 1, 1], 13: [0, 0, 0, 0, 1, 0], 14: [0, 0, 0, 0, 0, 1], 15: [0, 0, 0, 0, 0, 0], 16: [0, 0, -1, 1, 1, 1], 17: [0, 0, -1, 1, 1, 0], 18: [0, 0, -1, 1, 0, 1], 19: [0, 0, -1, 1, 0, 0], 20: [0, 0, -1, 0, 1, 1], 21: [0, 0, -1, 0, 1, 0], 22: [0, 0, -1, 0, 0, 1], 23: [0, 0, -1, 0, 0, 0], 24: [0, 0, 1, 1, 1, 1], 25: [0, 0, 1, 1, 1, 0], 26: [0, 0, 1, 1, 0, 1], 27: [0, 0, 1, 1, 0, 0], 28: [0, 0, 1, 0, 1, 1], 29: [0, 0, 1, 0, 1, 0], 30: [0, 0, 1, 0, 0, 1], 31: [0, 0, 1, 0, 0, 0], 32: [0, -1, 0, 1, 1, 1], 33: [0, -1, 0, 1, 1, 0], 34: [0, -1, 0, 1, 0, 1], 35: [0, -1, 0, 1, 0, 0], 36: [0, -1, 0, 0, 1, 1], 37: [0, -1, 0, 0, 1, 0], 38: [0, -1, 0, 0, 0, 1], 39: [0, -1, 0, 0, 0, 0], 40: [0, -1, -1, 1, 1, 1], 41: [0, -1, -1, 1, 1, 0], 42: [0, -1, -1, 1, 0, 1], 43: [0, -1, -1, 1, 0, 0], 44: [0, -1, -1, 0, 1, 1], 45: [0, -1, -1, 0, 1, 0], 46: [0, -1, -1, 0, 0, 1], 47: [0, -1, -1, 0, 0, 0], 48: [0, -1, 1, 1, 1, 1], 49: [0, -1, 1, 1, 1, 0], 50: [0, -1, 1, 1, 0, 1], 51: [0, -1, 1, 1, 0, 0], 52: [0, -1, 1, 0, 1, 1], 53: [0, -1, 1, 0, 1, 0], 54: [0, -1, 1, 0, 0, 1], 55: [0, -1, 1, 0, 0, 0], 56: [0, 1, 0, 1, 1, 1], 57: [0, 1, 0, 1, 1, 0], 58: [0, 1, 0, 1, 0, 1], 59: [0, 1, 0, 1, 0, 0], 60: [0, 1, 0, 0, 1, 1], 61: [0, 1, 0, 0, 1, 0], 62: [0, 1, 0, 0, 0, 1], 63: [0, 1, 0, 0, 0, 0], 64: [0, 1, -1, 1, 1, 1], 65: [0, 1, -1, 1, 1, 0], 66: [0, 1, -1, 1, 0, 1], 67: [0, 1, -1, 1, 0, 0], 68: [0, 1, -1, 0, 1, 1], 69: [0, 1, -1, 0, 1, 0], 70: [0, 1, -1, 0, 0, 1], 71: [0, 1, -1, 0, 0, 0], 72: [0, 1, 1, 1, 1, 1], 73: [0, 1, 1, 1, 1, 0], 74: [0, 1, 1, 1, 0, 1], 75: [0, 1, 1, 1, 0, 0], 76: [0, 1, 1, 0, 1, 1], 77: [0, 1, 1, 0, 1, 0], 78: [0, 1, 1, 0, 0, 1], 79: [0, 1, 1, 0, 0, 0], 80: [-1, 0, 0, 1, 1, 1], 81: [-1, 0, 0, 1, 1, 0], 82: [-1, 0, 0, 1, 0, 1], 83: [-1, 0, 0, 1, 0, 0], 84: [-1, 0, 0, 0, 1, 1], 85: [-1, 0, 0, 0, 1, 0], 86: [-1, 0, 0, 0, 0, 1], 87: [-1, 0, 0, 0, 0, 0], 88: [-1, 0, -1, 1, 1, 1], 89: [-1, 0, -1, 1, 1, 0], 90: [-1, 0, -1, 1, 0, 1], 91: [-1, 0, -1, 1, 0, 0], 92: [-1, 0, -1, 0, 1, 1], 93: [-1, 0, -1, 0, 1, 0], 94: [-1, 0, -1, 0, 0, 1], 95: [-1, 0, -1, 0, 0, 0], 96: [-1, 0, 1, 1, 1, 1], 97: [-1, 0, 1, 1, 1, 0], 98: [-1, 0, 1, 1, 0, 1], 99: [-1, 0, 1, 1, 0, 0], 100: [-1, 0, 1, 0, 1, 1], 101: [-1, 0, 1, 0, 1, 0], 102: [-1, 0, 1, 0, 0, 1], 103: [-1, 0, 1, 0, 0, 0], 104: [-1, -1, 0, 1, 1, 1], 105: [-1, -1, 0, 1, 1, 0], 106: [-1, -1, 0, 1, 0, 1], 107: [-1, -1, 0, 1, 0, 0], 108: [-1, -1, 0, 0, 1, 1], 109: [-1, -1, 0, 0, 1, 0], 110: [-1, -1, 0, 0, 0, 1], 111: [-1, -1, 0, 0, 0, 0], 112: [-1, -1, -1, 1, 1, 1], 113: [-1, -1, -1, 1, 1, 0], 114: [-1, -1, -1, 1, 0, 1], 115: [-1, -1, -1, 1, 0, 0], 116: [-1, -1, -1, 0, 1, 1], 117: [-1, -1, -1, 0, 1, 0], 118: [-1, -1, -1, 0, 0, 1], 119: [-1, -1, -1, 0, 0, 0], 120: [-1, -1, 1, 1, 1, 1], 121: [-1, -1, 1, 1, 1, 0], 122: [-1, -1, 1, 1, 0, 1], 123: [-1, -1, 1, 1, 0, 0], 124: [-1, -1, 1, 0, 1, 1], 125: [-1, -1, 1, 0, 1, 0], 126: [-1, -1, 1, 0, 0, 1], 127: [-1, -1, 1, 0, 0, 0], 128: [-1, 1, 0, 1, 1, 1], 129: [-1, 1, 0, 1, 1, 0], 130: [-1, 1, 0, 1, 0, 1], 131: [-1, 1, 0, 1, 0, 0], 132: [-1, 1, 0, 0, 1, 1], 133: [-1, 1, 0, 0, 1, 0], 134: [-1, 1, 0, 0, 0, 1], 135: [-1, 1, 0, 0, 0, 0], 136: [-1, 1, -1, 1, 1, 1], 137: [-1, 1, -1, 1, 1, 0], 138: [-1, 1, -1, 1, 0, 1], 139: [-1, 1, -1, 1, 0, 0], 140: [-1, 1, -1, 0, 1, 1], 141: [-1, 1, -1, 0, 1, 0], 142: [-1, 1, -1, 0, 0, 1], 143: [-1, 1, -1, 0, 0, 0], 144: [-1, 1, 1, 1, 1, 1], 145: [-1, 1, 1, 1, 1, 0], 146: [-1, 1, 1, 1, 0, 1], 147: [-1, 1, 1, 1, 0, 0], 148: [-1, 1, 1, 0, 1, 1], 149: [-1, 1, 1, 0, 1, 0], 150: [-1, 1, 1, 0, 0, 1], 151: [-1, 1, 1, 0, 0, 0], 152: [1, 0, 0, 1, 1, 1], 153: [1, 0, 0, 1, 1, 0], 154: [1, 0, 0, 1, 0, 1], 155: [1, 0, 0, 1, 0, 0], 156: [1, 0, 0, 0, 1, 1], 157: [1, 0, 0, 0, 1, 0], 158: [1, 0, 0, 0, 0, 1], 159: [1, 0, 0, 0, 0, 0], 160: [1, 0, -1, 1, 1, 1], 161: [1, 0, -1, 1, 1, 0], 162: [1, 0, -1, 1, 0, 1], 163: [1, 0, -1, 1, 0, 0], 164: [1, 0, -1, 0, 1, 1], 165: [1, 0, -1, 0, 1, 0], 166: [1, 0, -1, 0, 0, 1], 167: [1, 0, -1, 0, 0, 0], 168: [1, 0, 1, 1, 1, 1], 169: [1, 0, 1, 1, 1, 0], 170: [1, 0, 1, 1, 0, 1], 171: [1, 0, 1, 1, 0, 0], 172: [1, 0, 1, 0, 1, 1], 173: [1, 0, 1, 0, 1, 0], 174: [1, 0, 1, 0, 0, 1], 175: [1, 0, 1, 0, 0, 0], 176: [1, -1, 0, 1, 1, 1], 177: [1, -1, 0, 1, 1, 0], 178: [1, -1, 0, 1, 0, 1], 179: [1, -1, 0, 1, 0, 0], 180: [1, -1, 0, 0, 1, 1], 181: [1, -1, 0, 0, 1, 0], 182: [1, -1, 0, 0, 0, 1], 183: [1, -1, 0, 0, 0, 0], 184: [1, -1, -1, 1, 1, 1], 185: [1, -1, -1, 1, 1, 0], 186: [1, -1, -1, 1, 0, 1], 187: [1, -1, -1, 1, 0, 0], 188: [1, -1, -1, 0, 1, 1], 189: [1, -1, -1, 0, 1, 0], 190: [1, -1, -1, 0, 0, 1], 191: [1, -1, -1, 0, 0, 0], 192: [1, -1, 1, 1, 1, 1], 193: [1, -1, 1, 1, 1, 0], 194: [1, -1, 1, 1, 0, 1], 195: [1, -1, 1, 1, 0, 0], 196: [1, -1, 1, 0, 1, 1], 197: [1, -1, 1, 0, 1, 0], 198: [1, -1, 1, 0, 0, 1], 199: [1, -1, 1, 0, 0, 0], 200: [1, 1, 0, 1, 1, 1], 201: [1, 1, 0, 1, 1, 0], 202: [1, 1, 0, 1, 0, 1], 203: [1, 1, 0, 1, 0, 0], 204: [1, 1, 0, 0, 1, 1], 205: [1, 1, 0, 0, 1, 0], 206: [1, 1, 0, 0, 0, 1], 207: [1, 1, 0, 0, 0, 0], 208: [1, 1, -1, 1, 1, 1], 209: [1, 1, -1, 1, 1, 0], 210: [1, 1, -1, 1, 0, 1], 211: [1, 1, -1, 1, 0, 0], 212: [1, 1, -1, 0, 1, 1], 213: [1, 1, -1, 0, 1, 0], 214: [1, 1, -1, 0, 0, 1], 215: [1, 1, -1, 0, 0, 0]}
#dic_init_angle_summery={'(1,1,1)':0,'(0,0,0)':0,'(0,0,-1)':0,'(0,0,1)':0,'(0,-1,0)':0,'(0,-1,-1)':0,'(0,-1,1)':0,'(0,1,0)':0,'(0,1,-1)':0,'(0,1,1)':0,'(-1,0,0)':0,'(-1,0,-1)':0,'(-1,0,1)':0,'(-1,-1,0)':0,'(-1,-1,-1)':0,'(-1,-1,1)':0,'(-1,1,0)':0,'(-1,1,-1)':0,'(-1,1,1)':0,'(1,0,0)':0,'(1,0,-1)':0,'(1,0,1)':0,'(1,-1,0)':0,'(1,-1,-1)':0,'(1,-1,1)':0,'(1,1,0)':0,'(1,1,-1)':0}
# update : should add the AB(1)_AC(1)_BC(1) to the dic_action_index
dic_action_index = {0: 'THETA1_INC_THETA2_INC_THETA3_INC_AB1AC1BC1', 1: 'THETA1_INC_THETA2_INC_THETA3_INC_AB1AC1BC0', 2: 'THETA1_INC_THETA2_INC_THETA3_INC_AB1AC0BC1', 3: 'THETA1_INC_THETA2_INC_THETA3_INC_AB1AC0BC0', 4: 'THETA1_INC_THETA2_INC_THETA3_INC_AB0AC1BC1', 5: 'THETA1_INC_THETA2_INC_THETA3_INC_AB0AC1BC0', 6: 'THETA1_INC_THETA2_INC_THETA3_INC_AB0AC0BC1', 7: 'THETA1_INC_THETA2_INC_THETA3_INC_AB0AC0BC0', 8: 'THETA1_DN_THETA2_DN_THETA3_DN_AB1AC1BC1', 9: 'THETA1_DN_THETA2_DN_THETA3_DN_AB1AC1BC0', 10: 'THETA1_DN_THETA2_DN_THETA3_DN_AB1AC0BC1', 11: 'THETA1_DN_THETA2_DN_THETA3_DN_AB1AC0BC0', 12: 'THETA1_DN_THETA2_DN_THETA3_DN_AB0AC1BC1', 13: 'THETA1_DN_THETA2_DN_THETA3_DN_AB0AC1BC0', 14: 'THETA1_DN_THETA2_DN_THETA3_DN_AB0AC0BC1', 15: 'THETA1_DN_THETA2_DN_THETA3_DN_AB0AC0BC0', 16: 'THETA1_DN_THETA2_DN_THETA3_DEC_AB1AC1BC1', 17: 'THETA1_DN_THETA2_DN_THETA3_DEC_AB1AC1BC0', 18: 'THETA1_DN_THETA2_DN_THETA3_DEC_AB1AC0BC1', 19: 'THETA1_DN_THETA2_DN_THETA3_DEC_AB1AC0BC0', 20: 'THETA1_DN_THETA2_DN_THETA3_DEC_AB0AC1BC1', 21: 'THETA1_DN_THETA2_DN_THETA3_DEC_AB0AC1BC0', 22: 'THETA1_DN_THETA2_DN_THETA3_DEC_AB0AC0BC1', 23: 'THETA1_DN_THETA2_DN_THETA3_DEC_AB0AC0BC0', 24: 'THETA1_DN_THETA2_DN_THETA3_INC_AB1AC1BC1', 25: 'THETA1_DN_THETA2_DN_THETA3_INC_AB1AC1BC0', 26: 'THETA1_DN_THETA2_DN_THETA3_INC_AB1AC0BC1', 27: 'THETA1_DN_THETA2_DN_THETA3_INC_AB1AC0BC0', 28: 'THETA1_DN_THETA2_DN_THETA3_INC_AB0AC1BC1', 29: 'THETA1_DN_THETA2_DN_THETA3_INC_AB0AC1BC0', 30: 'THETA1_DN_THETA2_DN_THETA3_INC_AB0AC0BC1', 31: 'THETA1_DN_THETA2_DN_THETA3_INC_AB0AC0BC0', 32: 'THETA1_DN_THETA2_DEC_THETA3_DN_AB1AC1BC1', 33: 'THETA1_DN_THETA2_DEC_THETA3_DN_AB1AC1BC0', 34: 'THETA1_DN_THETA2_DEC_THETA3_DN_AB1AC0BC1', 35: 'THETA1_DN_THETA2_DEC_THETA3_DN_AB1AC0BC0', 36: 'THETA1_DN_THETA2_DEC_THETA3_DN_AB0AC1BC1', 37: 'THETA1_DN_THETA2_DEC_THETA3_DN_AB0AC1BC0', 38: 'THETA1_DN_THETA2_DEC_THETA3_DN_AB0AC0BC1', 39: 'THETA1_DN_THETA2_DEC_THETA3_DN_AB0AC0BC0', 40: 'THETA1_DN_THETA2_DEC_THETA3_DEC_AB1AC1BC1', 41: 'THETA1_DN_THETA2_DEC_THETA3_DEC_AB1AC1BC0', 42: 'THETA1_DN_THETA2_DEC_THETA3_DEC_AB1AC0BC1', 43: 'THETA1_DN_THETA2_DEC_THETA3_DEC_AB1AC0BC0', 44: 'THETA1_DN_THETA2_DEC_THETA3_DEC_AB0AC1BC1', 45: 'THETA1_DN_THETA2_DEC_THETA3_DEC_AB0AC1BC0', 46: 'THETA1_DN_THETA2_DEC_THETA3_DEC_AB0AC0BC1', 47: 'THETA1_DN_THETA2_DEC_THETA3_DEC_AB0AC0BC0', 48: 'THETA1_DN_THETA2_DEC_THETA3_INC_AB1AC1BC1', 49: 'THETA1_DN_THETA2_DEC_THETA3_INC_AB1AC1BC0', 50: 'THETA1_DN_THETA2_DEC_THETA3_INC_AB1AC0BC1', 51: 'THETA1_DN_THETA2_DEC_THETA3_INC_AB1AC0BC0', 52: 'THETA1_DN_THETA2_DEC_THETA3_INC_AB0AC1BC1', 53: 'THETA1_DN_THETA2_DEC_THETA3_INC_AB0AC1BC0', 54: 'THETA1_DN_THETA2_DEC_THETA3_INC_AB0AC0BC1', 55: 'THETA1_DN_THETA2_DEC_THETA3_INC_AB0AC0BC0', 56: 'THETA1_DN_THETA2_INC_THETA3_DN_AB1AC1BC1', 57: 'THETA1_DN_THETA2_INC_THETA3_DN_AB1AC1BC0', 58: 'THETA1_DN_THETA2_INC_THETA3_DN_AB1AC0BC1', 59: 'THETA1_DN_THETA2_INC_THETA3_DN_AB1AC0BC0', 60: 'THETA1_DN_THETA2_INC_THETA3_DN_AB0AC1BC1', 61: 'THETA1_DN_THETA2_INC_THETA3_DN_AB0AC1BC0', 62: 'THETA1_DN_THETA2_INC_THETA3_DN_AB0AC0BC1', 63: 'THETA1_DN_THETA2_INC_THETA3_DN_AB0AC0BC0', 64: 'THETA1_DN_THETA2_INC_THETA3_DEC_AB1AC1BC1', 65: 'THETA1_DN_THETA2_INC_THETA3_DEC_AB1AC1BC0', 66: 'THETA1_DN_THETA2_INC_THETA3_DEC_AB1AC0BC1', 67: 'THETA1_DN_THETA2_INC_THETA3_DEC_AB1AC0BC0', 68: 'THETA1_DN_THETA2_INC_THETA3_DEC_AB0AC1BC1', 69: 'THETA1_DN_THETA2_INC_THETA3_DEC_AB0AC1BC0', 70: 'THETA1_DN_THETA2_INC_THETA3_DEC_AB0AC0BC1', 71: 'THETA1_DN_THETA2_INC_THETA3_DEC_AB0AC0BC0', 72: 'THETA1_DN_THETA2_INC_THETA3_INC_AB1AC1BC1', 73: 'THETA1_DN_THETA2_INC_THETA3_INC_AB1AC1BC0', 74: 'THETA1_DN_THETA2_INC_THETA3_INC_AB1AC0BC1', 75: 'THETA1_DN_THETA2_INC_THETA3_INC_AB1AC0BC0', 76: 'THETA1_DN_THETA2_INC_THETA3_INC_AB0AC1BC1', 77: 'THETA1_DN_THETA2_INC_THETA3_INC_AB0AC1BC0', 78: 'THETA1_DN_THETA2_INC_THETA3_INC_AB0AC0BC1', 79: 'THETA1_DN_THETA2_INC_THETA3_INC_AB0AC0BC0', 80: 'THETA1_DEC_THETA2_DN_THETA3_DN_AB1AC1BC1', 81: 'THETA1_DEC_THETA2_DN_THETA3_DN_AB1AC1BC0', 82: 'THETA1_DEC_THETA2_DN_THETA3_DN_AB1AC0BC1', 83: 'THETA1_DEC_THETA2_DN_THETA3_DN_AB1AC0BC0', 84: 'THETA1_DEC_THETA2_DN_THETA3_DN_AB0AC1BC1', 85: 'THETA1_DEC_THETA2_DN_THETA3_DN_AB0AC1BC0', 86: 'THETA1_DEC_THETA2_DN_THETA3_DN_AB0AC0BC1', 87: 'THETA1_DEC_THETA2_DN_THETA3_DN_AB0AC0BC0', 88: 'THETA1_DEC_THETA2_DN_THETA3_DEC_AB1AC1BC1', 89: 'THETA1_DEC_THETA2_DN_THETA3_DEC_AB1AC1BC0', 90: 'THETA1_DEC_THETA2_DN_THETA3_DEC_AB1AC0BC1', 91: 'THETA1_DEC_THETA2_DN_THETA3_DEC_AB1AC0BC0', 92: 'THETA1_DEC_THETA2_DN_THETA3_DEC_AB0AC1BC1', 93: 'THETA1_DEC_THETA2_DN_THETA3_DEC_AB0AC1BC0', 94: 'THETA1_DEC_THETA2_DN_THETA3_DEC_AB0AC0BC1', 95: 'THETA1_DEC_THETA2_DN_THETA3_DEC_AB0AC0BC0', 96: 'THETA1_DEC_THETA2_DN_THETA3_INC_AB1AC1BC1', 97: 'THETA1_DEC_THETA2_DN_THETA3_INC_AB1AC1BC0', 98: 'THETA1_DEC_THETA2_DN_THETA3_INC_AB1AC0BC1', 99: 'THETA1_DEC_THETA2_DN_THETA3_INC_AB1AC0BC0', 100: 'THETA1_DEC_THETA2_DN_THETA3_INC_AB0AC1BC1', 101: 'THETA1_DEC_THETA2_DN_THETA3_INC_AB0AC1BC0', 102: 'THETA1_DEC_THETA2_DN_THETA3_INC_AB0AC0BC1', 103: 'THETA1_DEC_THETA2_DN_THETA3_INC_AB0AC0BC0', 104: 'THETA1_DEC_THETA2_DEC_THETA3_DN_AB1AC1BC1', 105: 'THETA1_DEC_THETA2_DEC_THETA3_DN_AB1AC1BC0', 106: 'THETA1_DEC_THETA2_DEC_THETA3_DN_AB1AC0BC1', 107: 'THETA1_DEC_THETA2_DEC_THETA3_DN_AB1AC0BC0', 108: 'THETA1_DEC_THETA2_DEC_THETA3_DN_AB0AC1BC1', 109: 'THETA1_DEC_THETA2_DEC_THETA3_DN_AB0AC1BC0', 110: 'THETA1_DEC_THETA2_DEC_THETA3_DN_AB0AC0BC1', 111: 'THETA1_DEC_THETA2_DEC_THETA3_DN_AB0AC0BC0', 112: 'THETA1_DEC_THETA2_DEC_THETA3_DEC_AB1AC1BC1', 113: 'THETA1_DEC_THETA2_DEC_THETA3_DEC_AB1AC1BC0', 114: 'THETA1_DEC_THETA2_DEC_THETA3_DEC_AB1AC0BC1', 115: 'THETA1_DEC_THETA2_DEC_THETA3_DEC_AB1AC0BC0', 116: 'THETA1_DEC_THETA2_DEC_THETA3_DEC_AB0AC1BC1', 117: 'THETA1_DEC_THETA2_DEC_THETA3_DEC_AB0AC1BC0', 118: 'THETA1_DEC_THETA2_DEC_THETA3_DEC_AB0AC0BC1', 119: 'THETA1_DEC_THETA2_DEC_THETA3_DEC_AB0AC0BC0', 120: 'THETA1_DEC_THETA2_DEC_THETA3_INC_AB1AC1BC1', 121: 'THETA1_DEC_THETA2_DEC_THETA3_INC_AB1AC1BC0', 122: 'THETA1_DEC_THETA2_DEC_THETA3_INC_AB1AC0BC1', 123: 'THETA1_DEC_THETA2_DEC_THETA3_INC_AB1AC0BC0', 124: 'THETA1_DEC_THETA2_DEC_THETA3_INC_AB0AC1BC1', 125: 'THETA1_DEC_THETA2_DEC_THETA3_INC_AB0AC1BC0', 126: 'THETA1_DEC_THETA2_DEC_THETA3_INC_AB0AC0BC1', 127: 'THETA1_DEC_THETA2_DEC_THETA3_INC_AB0AC0BC0', 128: 'THETA1_DEC_THETA2_INC_THETA3_DN_AB1AC1BC1', 129: 'THETA1_DEC_THETA2_INC_THETA3_DN_AB1AC1BC0', 130: 'THETA1_DEC_THETA2_INC_THETA3_DN_AB1AC0BC1', 131: 'THETA1_DEC_THETA2_INC_THETA3_DN_AB1AC0BC0', 132: 'THETA1_DEC_THETA2_INC_THETA3_DN_AB0AC1BC1', 133: 'THETA1_DEC_THETA2_INC_THETA3_DN_AB0AC1BC0', 134: 'THETA1_DEC_THETA2_INC_THETA3_DN_AB0AC0BC1', 135: 'THETA1_DEC_THETA2_INC_THETA3_DN_AB0AC0BC0', 136: 'THETA1_DEC_THETA2_INC_THETA3_DEC_AB1AC1BC1', 137: 'THETA1_DEC_THETA2_INC_THETA3_DEC_AB1AC1BC0', 138: 'THETA1_DEC_THETA2_INC_THETA3_DEC_AB1AC0BC1', 139: 'THETA1_DEC_THETA2_INC_THETA3_DEC_AB1AC0BC0', 140: 'THETA1_DEC_THETA2_INC_THETA3_DEC_AB0AC1BC1', 141: 'THETA1_DEC_THETA2_INC_THETA3_DEC_AB0AC1BC0', 142: 'THETA1_DEC_THETA2_INC_THETA3_DEC_AB0AC0BC1', 143: 'THETA1_DEC_THETA2_INC_THETA3_DEC_AB0AC0BC0', 144: 'THETA1_DEC_THETA2_DN_THETA3_INC_AB1AC1BC1', 145: 'THETA1_DEC_THETA2_DN_THETA3_INC_AB1AC1BC0', 146: 'THETA1_DEC_THETA2_DN_THETA3_INC_AB1AC0BC1', 147: 'THETA1_DEC_THETA2_DN_THETA3_INC_AB1AC0BC0', 148: 'THETA1_DEC_THETA2_DN_THETA3_INC_AB0AC1BC1', 149: 'THETA1_DEC_THETA2_DN_THETA3_INC_AB0AC1BC0', 150: 'THETA1_DEC_THETA2_DN_THETA3_INC_AB0AC0BC1', 151: 'THETA1_DEC_THETA2_DN_THETA3_INC_AB0AC0BC0', 152: 'THETA1_INC_THETA2_DN_THETA3_DN_AB1AC1BC1', 153: 'THETA1_INC_THETA2_DN_THETA3_DN_AB1AC1BC0', 154: 'THETA1_INC_THETA2_DN_THETA3_DN_AB1AC0BC1', 155: 'THETA1_INC_THETA2_DN_THETA3_DN_AB1AC0BC0', 156: 'THETA1_INC_THETA2_DN_THETA3_DN_AB0AC1BC1', 157: 'THETA1_INC_THETA2_DN_THETA3_DN_AB0AC1BC0', 158: 'THETA1_INC_THETA2_DN_THETA3_DN_AB0AC0BC1', 159: 'THETA1_INC_THETA2_DN_THETA3_DN_AB0AC0BC0', 160: 'THETA1_INC_THETA2_DN_THETA3_DEC_AB1AC1BC1', 161: 'THETA1_INC_THETA2_DN_THETA3_DEC_AB1AC1BC0', 162: 'THETA1_INC_THETA2_DN_THETA3_DEC_AB1AC0BC1', 163: 'THETA1_INC_THETA2_DN_THETA3_DEC_AB1AC0BC0', 164: 'THETA1_INC_THETA2_DN_THETA3_DEC_AB0AC1BC1', 165: 'THETA1_INC_THETA2_DN_THETA3_DEC_AB0AC1BC0', 166: 'THETA1_INC_THETA2_DN_THETA3_DEC_AB0AC0BC1', 167: 'THETA1_INC_THETA2_DN_THETA3_DEC_AB0AC0BC0', 168: 'THETA1_INC_THETA2_DN_THETA3_INC_AB1AC1BC1', 169: 'THETA1_INC_THETA2_DN_THETA3_INC_AB1AC1BC0', 170: 'THETA1_INC_THETA2_DN_THETA3_INC_AB1AC0BC1', 171: 'THETA1_INC_THETA2_DN_THETA3_INC_AB1AC0BC0', 172: 'THETA1_INC_THETA2_DN_THETA3_INC_AB0AC1BC1', 173: 'THETA1_INC_THETA2_DN_THETA3_INC_AB0AC1BC0', 174: 'THETA1_INC_THETA2_DN_THETA3_INC_AB0AC0BC1', 175: 'THETA1_INC_THETA2_DN_THETA3_INC_AB0AC0BC0', 176: 'THETA1_INC_THETA2_DEC_THETA3_DN_AB1AC1BC1', 177: 'THETA1_INC_THETA2_DEC_THETA3_DN_AB1AC1BC0', 178: 'THETA1_INC_THETA2_DEC_THETA3_DN_AB1AC0BC1', 179: 'THETA1_INC_THETA2_DEC_THETA3_DN_AB1AC0BC0', 180: 'THETA1_INC_THETA2_DEC_THETA3_DN_AB0AC1BC1', 181: 'THETA1_INC_THETA2_DEC_THETA3_DN_AB0AC1BC0', 182: 'THETA1_INC_THETA2_DEC_THETA3_DN_AB0AC0BC1', 183: 'THETA1_INC_THETA2_DEC_THETA3_DN_AB0AC0BC0', 184: 'THETA1_INC_THETA2_DEC_THETA3_DEC_AB1AC1BC1', 185: 'THETA1_INC_THETA2_DEC_THETA3_DEC_AB1AC1BC0', 186: 'THETA1_INC_THETA2_DEC_THETA3_DEC_AB1AC0BC1', 187: 'THETA1_INC_THETA2_DEC_THETA3_DEC_AB1AC0BC0', 188: 'THETA1_INC_THETA2_DEC_THETA3_DEC_AB0AC1BC1', 189: 'THETA1_INC_THETA2_DEC_THETA3_DEC_AB0AC1BC0', 190: 'THETA1_INC_THETA2_DEC_THETA3_DEC_AB0AC0BC1', 191: 'THETA1_INC_THETA2_DEC_THETA3_DEC_AB0AC0BC0', 192: 'THETA1_INC_THETA2_DEC_THETA3_INC_AB1AC1BC1', 193: 'THETA1_INC_THETA2_DEC_THETA3_INC_AB1AC1BC0', 194: 'THETA1_INC_THETA2_DEC_THETA3_INC_AB1AC0BC1', 195: 'THETA1_INC_THETA2_DEC_THETA3_INC_AB1AC0BC0', 196: 'THETA1_INC_THETA2_DEC_THETA3_INC_AB0AC1BC1', 197: 'THETA1_INC_THETA2_DEC_THETA3_INC_AB0AC1BC0', 198: 'THETA1_INC_THETA2_DEC_THETA3_INC_AB0AC0BC1', 199: 'THETA1_INC_THETA2_DEC_THETA3_INC_AB0AC0BC0', 200: 'THETA1_INC_THETA2_INC_THETA3_DN_AB1AC1BC1', 201: 'THETA1_INC_THETA2_INC_THETA3_DN_AB1AC1BC0', 202: 'THETA1_INC_THETA2_INC_THETA3_DN_AB1AC0BC1', 203: 'THETA1_INC_THETA2_INC_THETA3_DN_AB1AC0BC0', 204: 'THETA1_INC_THETA2_INC_THETA3_DN_AB0AC1BC1', 205: 'THETA1_INC_THETA2_INC_THETA3_DN_AB0AC1BC0', 206: 'THETA1_INC_THETA2_INC_THETA3_DN_AB0AC0BC1', 207: 'THETA1_INC_THETA2_INC_THETA3_DN_AB0AC0BC0', 208: 'THETA1_INC_THETA2_INC_THETA3_DEC_AB1AC1BC1', 209: 'THETA1_INC_THETA2_INC_THETA3_DEC_AB1AC1BC0', 210: 'THETA1_INC_THETA2_INC_THETA3_DEC_AB1AC0BC1', 211: 'THETA1_INC_THETA2_INC_THETA3_DEC_AB1AC0BC0', 212: 'THETA1_INC_THETA2_INC_THETA3_DEC_AB0AC1BC1', 213: 'THETA1_INC_THETA2_INC_THETA3_DEC_AB0AC1BC0', 214: 'THETA1_INC_THETA2_INC_THETA3_DEC_AB0AC0BC1', 215: 'THETA1_INC_THETA2_INC_THETA3_DEC_AB0AC0BC0'}
dic_episode_lenth={}
dic_summery_action={0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0, 39: 0, 40: 0, 41: 0, 42: 0, 43: 0, 44: 0, 45: 0, 46: 0, 47: 0, 48: 0, 49: 0, 50: 0, 51: 0, 52: 0, 53: 0, 54: 0, 55: 0, 56: 0, 57: 0, 58: 0, 59: 0, 60: 0, 61: 0, 62: 0, 63: 0, 64: 0, 65: 0, 66: 0, 67: 0, 68: 0, 69: 0, 70: 0, 71: 0, 72: 0, 73: 0, 74: 0, 75: 0, 76: 0, 77: 0, 78: 0, 79: 0, 80: 0, 81: 0, 82: 0, 83: 0, 84: 0, 85: 0, 86: 0, 87: 0, 88: 0, 89: 0, 90: 0, 91: 0, 92: 0, 93: 0, 94: 0, 95: 0, 96: 0, 97: 0, 98: 0, 99: 0, 100: 0, 101: 0, 102: 0, 103: 0, 104: 0, 105: 0, 106: 0, 107: 0, 108: 0, 109: 0, 110: 0, 111: 0, 112: 0, 113: 0, 114: 0, 115: 0, 116: 0, 117: 0, 118: 0, 119: 0, 120: 0, 121: 0, 122: 0, 123: 0, 124: 0, 125: 0, 126: 0, 127: 0, 128: 0, 129: 0, 130: 0, 131: 0, 132: 0, 133: 0, 134: 0, 135: 0, 136: 0, 137: 0, 138: 0, 139: 0, 140: 0, 141: 0, 142: 0, 143: 0, 144: 0, 145: 0, 146: 0, 147: 0, 148: 0, 149: 0, 150: 0, 151: 0, 152: 0, 153: 0, 154: 0, 155: 0, 156: 0, 157: 0, 158: 0, 159: 0, 160: 0, 161: 0, 162: 0, 163: 0, 164: 0, 165: 0, 166: 0, 167: 0, 168: 0, 169: 0, 170: 0, 171: 0, 172: 0, 173: 0, 174: 0, 175: 0, 176: 0, 177: 0, 178: 0, 179: 0, 180: 0, 181: 0, 182: 0, 183: 0, 184: 0, 185: 0, 186: 0, 187: 0, 188: 0, 189: 0, 190: 0, 191: 0, 192: 0, 193: 0, 194: 0, 195: 0, 196: 0, 197: 0, 198: 0, 199: 0, 200: 0, 201: 0, 202: 0, 203: 0, 204: 0, 205: 0, 206: 0, 207: 0, 208: 0, 209: 0, 210: 0, 211: 0, 212: 0, 213: 0, 214: 0, 215: 0}
dic_test_summery_action={0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0, 39: 0, 40: 0, 41: 0, 42: 0, 43: 0, 44: 0, 45: 0, 46: 0, 47: 0, 48: 0, 49: 0, 50: 0, 51: 0, 52: 0, 53: 0, 54: 0, 55: 0, 56: 0, 57: 0, 58: 0, 59: 0, 60: 0, 61: 0, 62: 0, 63: 0, 64: 0, 65: 0, 66: 0, 67: 0, 68: 0, 69: 0, 70: 0, 71: 0, 72: 0, 73: 0, 74: 0, 75: 0, 76: 0, 77: 0, 78: 0, 79: 0, 80: 0, 81: 0, 82: 0, 83: 0, 84: 0, 85: 0, 86: 0, 87: 0, 88: 0, 89: 0, 90: 0, 91: 0, 92: 0, 93: 0, 94: 0, 95: 0, 96: 0, 97: 0, 98: 0, 99: 0, 100: 0, 101: 0, 102: 0, 103: 0, 104: 0, 105: 0, 106: 0, 107: 0, 108: 0, 109: 0, 110: 0, 111: 0, 112: 0, 113: 0, 114: 0, 115: 0, 116: 0, 117: 0, 118: 0, 119: 0, 120: 0, 121: 0, 122: 0, 123: 0, 124: 0, 125: 0, 126: 0, 127: 0, 128: 0, 129: 0, 130: 0, 131: 0, 132: 0, 133: 0, 134: 0, 135: 0, 136: 0, 137: 0, 138: 0, 139: 0, 140: 0, 141: 0, 142: 0, 143: 0, 144: 0, 145: 0, 146: 0, 147: 0, 148: 0, 149: 0, 150: 0, 151: 0, 152: 0, 153: 0, 154: 0, 155: 0, 156: 0, 157: 0, 158: 0, 159: 0, 160: 0, 161: 0, 162: 0, 163: 0, 164: 0, 165: 0, 166: 0, 167: 0, 168: 0, 169: 0, 170: 0, 171: 0, 172: 0, 173: 0, 174: 0, 175: 0, 176: 0, 177: 0, 178: 0, 179: 0, 180: 0, 181: 0, 182: 0, 183: 0, 184: 0, 185: 0, 186: 0, 187: 0, 188: 0, 189: 0, 190: 0, 191: 0, 192: 0, 193: 0, 194: 0, 195: 0, 196: 0, 197: 0, 198: 0, 199: 0, 200: 0, 201: 0, 202: 0, 203: 0, 204: 0, 205: 0, 206: 0, 207: 0, 208: 0, 209: 0, 210: 0, 211: 0, 212: 0, 213: 0, 214: 0, 215: 0}


dic_summery_steer_action={0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0}
dic_action_steer_index = {
"THETA1_INC_THETA2_INC_THETA3_INC":0 ,
"THETA1_DN_THETA2_DN_THETA3_DN"   :0 ,
"THETA1_DN_THETA2_DN_THETA3_DEC"  :0 ,
"THETA1_DN_THETA2_DN_THETA3_INC"  :0 ,
"THETA1_DN_THETA2_DEC_THETA3_DN"  :0 ,
"THETA1_DN_THETA2_DEC_THETA3_DEC" :0 ,
"THETA1_DN_THETA2_DEC_THETA3_INC" :0 ,
"THETA1_DN_THETA2_INC_THETA3_DN"  :0 ,
"THETA1_DN_THETA2_INC_THETA3_DEC" :0 ,
"THETA1_DN_THETA2_INC_THETA3_INC" :0 ,
"THETA1_DEC_THETA2_DN_THETA3_DN"  :0,
"THETA1_DEC_THETA2_DN_THETA3_DEC" :0,
"THETA1_DEC_THETA2_DN_THETA3_INC" :0,
"THETA1_DEC_THETA2_DEC_THETA3_DN" :0,
"THETA1_DEC_THETA2_DEC_THETA3_DEC":0,
"THETA1_DEC_THETA2_DEC_THETA3_INC":0,
"THETA1_DEC_THETA2_INC_THETA3_DN" :0,
"THETA1_DEC_THETA2_INC_THETA3_DEC":0,
"THETA1_DEC_THETA2_DN_THETA3_INC" :0,
"THETA1_INC_THETA2_DN_THETA3_DN"  :0,
"THETA1_INC_THETA2_DN_THETA3_DEC" :0,
"THETA1_INC_THETA2_DN_THETA3_INC" :0,
"THETA1_INC_THETA2_DEC_THETA3_DN" :0,
"THETA1_INC_THETA2_DEC_THETA3_DEC":0,
"THETA1_INC_THETA2_DEC_THETA3_INC":0,
"THETA1_INC_THETA2_INC_THETA3_DN" :0,
"THETA1_INC_THETA2_INC_THETA3_DEC":0}
STATE_TERMINAL = np.zeros((3,image_width,image_height))#[0,0,0,0,0,0,0,0,0]#"TERMINAL"
# these upper and lower bound angles represent the minimum and maximum streering angles for each camera
THETA1_UPPER_BOUND_ANGLE = 90
THETA2_UPPER_BOUND_ANGLE = 90
THETA3_UPPER_BOUND_ANGLE = 90
THETA1_LOWER_BOUND_ANGLE = 0
THETA2_LOWER_BOUND_ANGLE = 0
THETA3_LOWER_BOUND_ANGLE = 0

ssd_model = init_colab_model()
ssd_model2=init_ref_model()
ssd_model4=init_ref_complete_model()
ssd_model5= init_colab_complete_model()
ssd_model3 = init_ref_combinedLast_model()

#def normalize_image(image):
#    # Create an instance of ImagePreprocessor
#    preprocessor = ImagePreprocessor(observation_shape=(100, 100, 3), normalize=True)
#
#    # Normalize the image
#    normalized_image = preprocessor.transform(image)
#
#    return normalized_image

def process_chunk(chunk):
    local_dic = {}
    for line in chunk:
        line = line.strip()
        parts = line.split(": ")
        filename = parts[0]
        bounding_boxes = ast.literal_eval(parts[1])
        local_dic[filename] = bounding_boxes
    return local_dic
def parallel_read_dic(file_path, num_processes):
    # Read the file and split lines into chunks
    with open(file_path, "r") as file:
        lines = file.readlines()

    chunk_size = len(lines) // num_processes
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

    # Create a multiprocessing Pool
    pool = mp.Pool(processes=num_processes)

    # Use pool.map to process chunks in parallel
    result_dicts = pool.map(process_chunk, chunks)

    # Combine the results from different processes into one dictionary
    dic_accuracy_state={}
    for result in result_dicts:
        dic_accuracy_state.update(result)
    # Close the pool
    pool.close()
    pool.join()
    return dic_accuracy_state
def read_image(file_path):
    new_file_path = './YOLOV3/'+dataset_load_name
    file_path = os.path.join(new_file_path,file_path)
    image = cv2.imread(file_path)
    return file_path, image

def parallel_read_images(folder_path, num_processes):
    image_list = {}
    print("loading the images started ")
    file_paths = []  # Store file paths and image data as tuples
    for char in ['A']:#, 'B', 'C']:
        for i in [0]:#, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
            for image_id in range(500):
                if char == 'A':
                    view = "1"
                elif char == 'B':
                    view = "2"
                elif char == 'C':
                    view = "3"
                frameID=image_id
                if(len(str(frameID))==1):
                  filenamex =  "frame_000"+str(frameID)+".jpg"
                elif(len(str(frameID))==2):
                 filenamex = "frame_00"+str(frameID)+".jpg"
                else:
                 filenamex =  "frame_0"+str(frameID)+".jpg"
                file_name = f"{char}{i}/View_00{view}/"+filenamex
                #file_path = os.path.join(folder_path, file_name)
                #print(file_path)
                #if os.path.exists(file_path):
                file_paths.append(file_name)
    with Pool(processes=num_processes) as pool:
        results = pool.map(read_image, file_paths)

    # Collect the results into the image_list dictionary
    for file_path, image in results:
        if image is not None:
            image_list[file_path] = image
        else:
            print(f"Failed to load file '{file_path}' with OpenCV.")
    print("tttttttttttttttt")
    pdb.set_trace()
    return image_list

#image_dic = parallel_read_images('',80)
#print("ooooooooooooooooooooooo",image_dic)
# Now 'image_list' contains all the loaded images as OpenCV NumPy arrays

def parallel_read_dic_mAP(file_path, num_processes):
    # Read the file and split lines into chunks
    with open(file_path, "r") as file:
        lines = file.readlines()

    chunk_size = len(lines) // num_processes
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

    # Create a multiprocessing Pool
    pool = mp.Pool(processes=num_processes)

    # Use pool.map to process chunks in parallel
    result_dicts = pool.map(process_chunk, chunks)

    # Combine the results from different processes into one dictionary
    dic_mAP_accuracy_state={}
    for result in result_dicts:
        dic_mAP_accuracy_state.update(result)
    # Close the pool
    pool.close()
    pool.join()
    return dic_mAP_accuracy_state
def parallel_read_dic_TPFPFN(file_path, num_processes):
    # Define the function to process a chunk of lines

    # Read the file and split lines into chunks
    with open(file_path, "r") as file:
        lines = file.readlines()

    chunk_size = len(lines) // num_processes
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

    # Create a multiprocessing Pool
    pool = mp.Pool(processes=num_processes)

    # Use pool.map to process chunks in parallel
    result_dicts = pool.map(process_chunk, chunks)

    # Combine the results from different processes into one dictionary
    dic_TPFPFN_accuracy_state={}
    for result in result_dicts:
        dic_TPFPFN_accuracy_state.update(result)
    # Close the pool
    pool.close()
    pool.join()
    return dic_TPFPFN_accuracy_state
dic_detection_results={}
dic_mAP_results={}
dic_TPFPFN_results={}
data_suppix_list = ["A_B_C", "A_B","A_C", "A","B_A_C","B_A","B_C","B","C_A_B","C_A","C_B", "C"]
start_time = time.time() 
#for f in data_suppix_list:
#  load_path_detection= "./out/new_detection_results"+f+"comAI.txt"
#  load_path_mAP = "./out/new_detection_mAP_results_"+f+"comAI.txt"
#  load_path_TPFPFN = "./out/new_detection_TPFPFN_results_"+f+"comAI.txt"
#  dic_detection_results[f] = parallel_read_dic(load_path_detection,80)
#  dic_mAP_results[f] = parallel_read_dic_mAP(load_path_mAP,80)
#  dic_TPFPFN_results[f] =parallel_read_dic_TPFPFN(load_path_TPFPFN,80)
file_path1 = "./full_new_detection_results.json"  # Replace with your desired file path
file_path2 = "./full_new_detection_mAP_results.json"
file_path3 = "./full_new_detection_TPFPFN_results.json"
#file_path1 = "./full_new_detection_results_Tjunction.json"  # Replace with your desired file path
#file_path2 = "./full_new_detection_mAP_results_Tjunction.json"
#file_path3 = "./full_new_detection_TPFPFN_results_Tjunction.json"
with open(file_path1, 'r') as file1:
  dic_detection_results = json.load(file1)
with open(file_path2, 'r') as file2:
  dic_mAP_results = json.load(file2)
with open(file_path3, 'r') as file3:
  dic_TPFPFN_results = json.load(file3)
print("############################################time taken for reading the files is : ", (time.time() - start_time) * 1000, " ms")

def normalize_image(image):
    # Scale pixel values to the range [0, 1]
    #image = image.astype(np.float32) / 255.0

    # Apply channel-wise normalization
    #mean = np.mean(image, axis=(0, 1))
    #std = np.std(image, axis=(0, 1))
    #image = (image - mean) / std
    resized_image = cv2.resize(image[0,:,:,:], (150, 150))
    # Convert the resized image to grayscale 
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    return grayscale_image



class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
 
    def __init__(self,env, verbose=0):
        super().__init__(verbose)
        self.env=env
    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        #value = np.random.random()
        #dic_summery_action[action]+=1
        
        currentaction= dic_action_index[self.env.action_totake]
        dic_action_steer_index[currentaction[:-10]]+=1
        #print(currentaction,currentaction[:-10])
        self.logger.record("action/"+currentaction[:-10], dic_action_steer_index[currentaction[:-10]])
        #self.logger.record("actions/1", dic_summery_action[1])
        #self.logger.record("actions/2", dic_summery_action[2])
        #self.logger.record("actions/3", dic_summery_action[3])
        #self.logger.record("actions/4", dic_summery_action[4])
        #self.logger.record("actions/5", dic_summery_action[5])
        #self.logger.record("actions/6", dic_summery_action[6])
        #self.logger.record("actions/7", dic_summery_action[7])
        #self.logger.record("actions/8", dic_summery_action[8])
        #self.logger.record("actions/9", dic_summery_action[9])
        #self.logger.record("actions/10", dic_summery_action[10])
        #self.logger.record("actions/11", dic_summery_action[11])
        #self.logger.record("actions/12", dic_summery_action[12])
        #self.logger.record("actions/13", dic_summery_action[13])
        #self.logger.record("actions/14", dic_summery_action[14])
        #self.logger.record("actions/15", dic_summery_action[15])
        #self.logger.record("actions/16", dic_summery_action[16])
        #self.logger.record("actions/17", dic_summery_action[17])
        #self.logger.record("actions/18", dic_summery_action[18])
        #self.logger.record("actions/19", dic_summery_action[19])
        #self.logger.record("actions/20", dic_summery_action[20])
        #self.logger.record("actions/21", dic_summery_action[21])
        #self.logger.record("actions/22", dic_summery_action[22])
        #self.logger.record("actions/23", dic_summery_action[23])
        #self.logger.record("actions/24", dic_summery_action[24])
        #self.logger.record("actions/25", dic_summery_action[25])
        #self.logger.record("actions/26", dic_summery_action[26])
     
        return True


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    def __init__(self, traning=False):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        N_DISCRETE_ACTIONS = 27*8
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input:
        #N_CHANNELS=9
        self.observation_space = spaces.Box(low=0, high=255,shape=(3,image_width,image_height), dtype=np.uint8)
        self.player1 = None
        self.training=traning
        self.THETA1=0
        self.THETA2=0
        self.THETA3=0
        self.AB=0
        self.AC=0
        self.BC=0
        self.done= False
        self.random_action =0
        self.c1c2_area=0
        self.c1c3_area=0
        self.c2c3_area=0
        self.c1c2_people=0
        self.c1c3_people=0
        self.c2c3_people=0
        self.c1c2_occlusion=0
        self.c1c3_occlusion=0
        self.c2c3_occlusion=0
        self.current_iteration=0
        self.pre_v1_acuracy=0
        self.pre_v2_acuracy=0
        self.pre_v3_acuracy=0
        self.yolo=None
        self.current_state_number=0 
        self.alpha = 2.2#0.6 # 0.8 
        self.beta = 0.2#0.4 #0.2
        self.T=10
        #pdb.set_trace()
        self.dic_accuracy_state={}
        self.dic_mAP_accuracy_state={}
        self.dic_TPFPFN_accuracy_state={}
        self.dic_coverage_accuracy_state={}
        self.dic_transformation_state ={}
        self.p1=0
        self.p2=0
        self.p3=0
        self.action_totake=0
        #self.dic_images=image_dic
        
        print("----------------loading large file started---------------")
        #self.read_dic()
        #self.read_dic_mAP()
        self.en_from_text=0
        
        self.read_dic_coverage()
        #self.read_dic_TPFPFN()
        self.read_dic_transformation()
        print("loading large file ended")
    
        self.seed=2
    def read_dic_transformation(self):
      with open("./index_transformation_data.txt", "r") as file:
         lines = file.readlines()
      for line in lines:
         line = eval(line)
         dic_key= line.keys()
         str_dic_key=str(list(dic_key)[0])
         dic_a=line[str_dic_key]
         self.dic_transformation_state[str_dic_key]=dic_a
    def read_dic(self):
      with open("./all_detection_resultscomAI.txt", "r") as file:
         lines = file.readlines()
      for line in lines:
          line = line.strip()  # Remove leading/trailing whitespaces
          parts = line.split(": ")  # Split the line into filename and bounding boxes
          filename = parts[0]
          bounding_boxes = parts[1]
          bounding_boxes = ast.literal_eval(bounding_boxes)
          self.dic_accuracy_state[filename]=bounding_boxes

    def read_dic_mAP(self):
      with open("./all_detection_mAP_results"+data_suppix+".txt", "r") as file:
         lines = file.readlines()
      for line in lines:
          line = line.strip()  # Remove leading/trailing whitespaces
          parts = line.split(": ")  # Split the line into filename and bounding boxes
          filename = parts[0]
          bounding_boxes = parts[1]
          bounding_boxes = ast.literal_eval(bounding_boxes)
          self.dic_mAP_accuracy_state[filename]=bounding_boxes
    def read_dic_TPFPFN(self):
      with open("./all_detection_TPFPFN_results"+data_suppix+".txt", "r") as file:
         lines = file.readlines()
      for line in lines:
          line = line.strip()  # Remove leading/trailing whitespaces
          parts = line.split(": ")  # Split the line into filename and bounding boxes
          filename = parts[0]
          bounding_boxes = parts[1]
          bounding_boxes = ast.literal_eval(bounding_boxes)
          self.dic_TPFPFN_accuracy_state[filename]=bounding_boxes
    def read_dic_coverage(self):
      #with open("./camera_angles_vs_covring_area.txt", "r") as file:
      with open("./camera_angles_vs_covring_area.txt", 'r') as file:
         data = json.load(file)
      for key, value in data.items():
          self.dic_coverage_accuracy_state[key] = value
  

    def seed(self, seed=None):
        return 2

    def preprocess_image(self, c1_view, c2_view, c3_view):
        stacked_images = np.stack((c1_view, c2_view, c3_view), axis=-1)
        
        #when the image is black and white no need to take the mean
        #averaged_image = np.mean(stacked_images, axis=-1)
        averaged_image=stacked_images.astype(np.uint8)
        averaged_image = np.transpose(averaged_image, (2, 0, 1))
        return averaged_image
    def step(self, action):
        #start_time1 = time.time() 
        #start_time = time.time()
        # ACTION_@1_INC,ACTION_@1_DEC,ACTION_@2_INC,ACTION_@2_DEC,ACTION_@3_INC,ACTION_@3_DEC,DO_NOTHING
        #Setof actions valid for the agnet. here the agent can chnage angle of cameras as it has one degree of freedom
        if(ENABLE_PRINT):
          print("step function has been called-------self.current_state_number, self.current_iteration ",self.current_state_number,self.current_iteration)
        action_list=dic_action[action] # action is type of  dic_action = 0: [1, 1, 1, 1, 1, 1] first three indices are for steering angles and last three are for collaboration
        str_exe_action = dic_action_index[action]
        self.action_totake = action

        A_x_old = "A"+str(self.THETA1)
        B_x_old = "B"+str(self.THETA2)
        C_x_old = "C"+str(self.THETA3)

        
        self.THETA1 = self.THETA1 +10*action_list[0]
        self.THETA2 = self.THETA2 +10*action_list[1]
        self.THETA3 = self.THETA3 +10*action_list[2]

        #these three bits corresponds to selecting the benifitial collaborators 
        self.AB = action_list[3]
        self.AC = action_list[4]
        self.BC = action_list[5]
        #print("executed action is ",str_exe_action,self.current_iteration)
        
        dic_summery_action[action]+=1
        #logger.record("rollout/exploration_rate", self.exploration_rate)
        self.current_state_number+= self.T
        value = self.current_state_number
        self.current_iteration= self.current_state_number
        A_x = "A"+str(self.THETA1)
        B_x = "B"+str(self.THETA2)
        C_x = "C"+str(self.THETA3)

        #fps = (time.time() - start_time)*1000
       # print(f"1111FPS: {fps:.2f} ms")
        #print("angles are",A_x,B_x,C_x)
        #print(self.THETA1 >= THETA1_LOWER_BOUND_ANGLE , self.THETA1 <= THETA1_UPPER_BOUND_ANGLE , self.THETA2 >= THETA2_LOWER_BOUND_ANGLE , self.THETA2 <= THETA2_UPPER_BOUND_ANGLE , self.THETA3 >= THETA3_LOWER_BOUND_ANGLE , self.THETA3 <= THETA3_UPPER_BOUND_ANGLE , self.current_iteration , MAx_ITER_PER_EPISODE )
        if (self.THETA1 >= THETA1_LOWER_BOUND_ANGLE and self.THETA1 <= THETA1_UPPER_BOUND_ANGLE and self.THETA2 >= THETA2_LOWER_BOUND_ANGLE and self.THETA2 <= THETA2_UPPER_BOUND_ANGLE and self.THETA3 >= THETA3_LOWER_BOUND_ANGLE and self.THETA3 <= THETA3_UPPER_BOUND_ANGLE and self.current_iteration < MAx_ITER_PER_EPISODE ): 
              start_time = time.time() 
              # self.get_people function should now consider accuracies in differant conditions. for and example. for a given
              # time instance "f" and view A now there 3 instance to consider depending on collaboration.
              #if best action for benificial collaborator is (AB,AC,BC) = (1,0,0) then A only colab with B and A's accuracy is selcted with obly one collaboraotr
              #if best action for benificial collaborator is (AB,AC,BC) = (0,1,0) then A only colab with C and A's accuracy is selcted with obly one collaboraotr
              #if best action for benificial collaborator is (AB,AC,BC) = (0,0,0) then A do not colab and A's accuracy is selcted without collboration
              #without taking the action
              
              #_,det_content1_old = self.get_people(view1= 1,view2= 2,view3= 3,value=value,ref=A_x_old,colab1=B_x_old,colab2=C_x_old,colab_action=[self.AB,self.AC,self.BC])
              #_,det_content2_old = self.get_people(view1= 2,view2= 1,view3= 3,value=value,ref=B_x_old,colab1=A_x_old,colab2=C_x_old,colab_action=[self.AB,self.AC,self.BC])
              #_,det_content3_old = self.get_people(view1= 3,view2= 1,view3= 2,value=value,ref=C_x_old,colab1=A_x_old,colab2=B_x_old,colab_action=[self.AB,self.AC,self.BC])
              ##OLD CODE
              #_,det_content1_old = self.get_people_base(view1= 1,view2= 2,view3= 3,value=value,ref=A_x_old,colab1=B_x_old,colab2=C_x_old,colab_action=[0,0,0])
              #_,det_content2_old = self.get_people_base(view1= 2,view2= 1,view3= 3,value=value,ref=B_x_old,colab1=A_x_old,colab2=C_x_old,colab_action=[0,0,0])
              #_,det_content3_old = self.get_people_base(view1= 3,view2= 1,view3= 2,value=value,ref=C_x_old,colab1=A_x_old,colab2=B_x_old,colab_action=[0,0,0])
              ##fps = (time.time() - start_time)*1000
              ##print(f"2222FPS: {fps:.2f} ms")
              #
              #with taking the action.
              #self.c1_view,det_content1 = self.get_people(view1= 1,view2= 2,view3= 3,value=value,ref=A_x,colab1=B_x,colab2=C_x,colab_action=[self.AB,self.AC,self.BC])
              #self.c2_view,det_content2 = self.get_people(view1= 2,view2= 1,view3= 3,value=value,ref=B_x,colab1=A_x,colab2=C_x,colab_action=[self.AB,self.AC,self.BC])
              #self.c3_view,det_content3 = self.get_people(view1= 3,view2= 1,view3= 2,value=value,ref=C_x,colab1=A_x,colab2=B_x,colab_action=[self.AB,self.AC,self.BC])
              ##start_time = time.time()  
              #mAP1= self.get_mAP(view =1,image_id=value,det_content=det_content1,ref=A_x,colab1=B_x,colab2=C_x,colab_action=[self.AB,self.AC,self.BC])
              #mAP2= self.get_mAP(view =2,image_id=value,det_content=det_content2,ref=B_x,colab1=A_x,colab2=C_x,colab_action=[self.AB,self.AC,self.BC])
              #mAP3= self.get_mAP(view =3,image_id=value,det_content=det_content3,ref=C_x,colab1=A_x,colab2=B_x,colab_action=[self.AB,self.AC,self.BC])
              ##fps = (time.time() - start_time)*1000
              ##print(f"33333FPS: {fps:.2f} ms")
              #mAP1_old= self.get_mAP(view =1,image_id=value,det_content=det_content1_old,ref=A_x_old,colab1=B_x_old,colab2=C_x_old,colab_action=[0,0,0])
              #mAP2_old= self.get_mAP(view =2,image_id=value,det_content=det_content2_old,ref=B_x_old,colab1=A_x_old,colab2=C_x_old,colab_action=[0,0,0])
              #mAP3_old= self.get_mAP(view =3,image_id=value,det_content=det_content3_old,ref=C_x_old,colab1=A_x_old,colab2=B_x_old,colab_action=[0,0,0])
              #  by taking the  differance here i consier by taking the action to chnage the angle of the camera and collborator pairs is it better than 

              #
              cummulative_total_colab=0
              cum_steer_total_diff = 0
              avg_diff_v1 =0
              avg_diff_v2 =0
              avg_diff_v3 =0
              avg_colab_diff_v1 = 0
              avg_colab_diff_v2 = 0
              avg_colab_diff_v3 = 0
              self.c1_view = cam_images[A_x+"_"+str(value)]#self.get_people(view1= 1,view2= 2,view3= 3,value=value,ref=A_x,colab1=B_x,colab2=C_x,colab_action=[self.AB,self.AC,self.BC])
              self.c2_view = cam_images[B_x+"_"+str(value)]#self.get_people(view1= 2,view2= 1,view3= 3,value=value,ref=B_x,colab1=A_x,colab2=C_x,colab_action=[self.AB,self.AC,self.BC])
              self.c3_view = cam_images[C_x+"_"+str(value)]#self.get_people(view1= 3,view2= 1,view3= 2,value=value,ref=C_x,colab1=A_x,colab2=B_x,colab_action=[self.AB,self.AC,self.BC])
              #self.BC=0
              #print(self.AB,self.BC,self.AC)
              for I in range(value-self.T+4,value):#  here 4 is i assume this much time is required to change the angle of the camera 
                _,det_content1_old = self.get_people_base(view1= 1,view2= 2,view3= 3,value=I,ref=A_x_old,colab1=B_x_old,colab2=C_x_old,colab_action=[self.AB,self.AC,self.BC])
                _,det_content2_old = self.get_people_base(view1= 2,view2= 1,view3= 3,value=I,ref=B_x_old,colab1=A_x_old,colab2=C_x_old,colab_action=[self.AB,self.AC,self.BC])
                _,det_content3_old = self.get_people_base(view1= 3,view2= 1,view3= 2,value=I,ref=C_x_old,colab1=A_x_old,colab2=B_x_old,colab_action=[self.AB,self.AC,self.BC])

                _,det_content2_new_0 = self.get_people_base(view1= 2,view2= 1,view3= 3,value=I,ref=B_x,colab1=A_x,colab2=C_x,colab_action=[0,0,0])
                _,det_content1_new_0 = self.get_people_base(view1= 1,view2= 2,view3= 3,value=I,ref=A_x,colab1=B_x,colab2=C_x,colab_action=[0,0,0])
                _,det_content3_new_0 = self.get_people_base(view1= 3,view2= 1,view3= 2,value=I,ref=C_x,colab1=A_x,colab2=B_x,colab_action=[0,0,0])
                #with taking the action.
                _,det_content1 = self.get_people_base(view1= 1,view2= 2,view3= 3,value=I,ref=A_x,colab1=B_x,colab2=C_x,colab_action=[self.AB,self.AC,self.BC])
                _,det_content2 = self.get_people_base(view1= 2,view2= 1,view3= 3,value=I,ref=B_x,colab1=A_x,colab2=C_x,colab_action=[self.AB,self.AC,self.BC])
                _,det_content3 = self.get_people_base(view1= 3,view2= 1,view3= 2,value=I,ref=C_x,colab1=A_x,colab2=B_x,colab_action=[self.AB,self.AC,self.BC])

                ###### below code is to calculate steering reward
                #start_time = time.time()  
                mAP1_withC= self.get_mAP(view =1,image_id=I,det_content=det_content1,ref=A_x,colab1=B_x,colab2=C_x,colab_action=[self.AB,self.AC,self.BC])
                mAP2_withC= self.get_mAP(view =2,image_id=I,det_content=det_content2,ref=B_x,colab1=A_x,colab2=C_x,colab_action=[self.AB,self.AC,self.BC])
                mAP3_withC= self.get_mAP(view =3,image_id=I,det_content=det_content3,ref=C_x,colab1=A_x,colab2=B_x,colab_action=[self.AB,self.AC,self.BC])

                mAP1= self.get_mAP(view =1,image_id=I,det_content=det_content1,ref=A_x,colab1=B_x,colab2=C_x,colab_action=[0,0,0])
                mAP2= self.get_mAP(view =2,image_id=I,det_content=det_content2,ref=B_x,colab1=A_x,colab2=C_x,colab_action=[0,0,0])
                mAP3= self.get_mAP(view =3,image_id=I,det_content=det_content3,ref=C_x,colab1=A_x,colab2=B_x,colab_action=[0,0,0])
                #fps = (time.time() - start_time)*1000
                #print(f"33333FPS: {fps:.2f} ms")
                #mAP1_old= self.get_mAP(view =1,image_id=I,det_content=det_content1_old,ref=A_x_old,colab1=B_x_old,colab2=C_x_old,colab_action=[self.AB,self.AC,self.BC])
                #mAP2_old= self.get_mAP(view =2,image_id=I,det_content=det_content2_old,ref=B_x_old,colab1=A_x_old,colab2=C_x_old,colab_action=[self.AB,self.AC,self.BC])
                #mAP3_old= self.get_mAP(view =3,image_id=I,det_content=det_content3_old,ref=C_x_old,colab1=A_x_old,colab2=B_x_old,colab_action=[self.AB,self.AC,self.BC])
                mAP1_old= self.get_mAP(view =1,image_id=I,det_content=det_content1_old,ref=A_x_old,colab1=B_x_old,colab2=C_x_old,colab_action=[0,0,0])
                mAP2_old= self.get_mAP(view =2,image_id=I,det_content=det_content2_old,ref=B_x_old,colab1=A_x_old,colab2=C_x_old,colab_action=[0,0,0])
                mAP3_old= self.get_mAP(view =3,image_id=I,det_content=det_content3_old,ref=C_x_old,colab1=A_x_old,colab2=B_x_old,colab_action=[0,0,0])
                
                # below differance corresponds to only instance where only form steering camera angle is changed
                diff_v1 = mAP1 - mAP1_old 
                diff_v2 = mAP2 - mAP2_old 
                diff_v3 = mAP3 - mAP3_old 

                mAP1_new_0= self.get_mAP(view =1,image_id=I,det_content=det_content1_new_0,ref=A_x,colab1=B_x,colab2=C_x,colab_action=[0,0,0])
                mAP2_new_0= self.get_mAP(view =2,image_id=I,det_content=det_content2_new_0,ref=B_x,colab1=A_x,colab2=C_x,colab_action=[0,0,0])
                mAP3_new_0= self.get_mAP(view =3,image_id=I,det_content=det_content3_new_0,ref=C_x,colab1=A_x,colab2=B_x,colab_action=[0,0,0])
                
                # below is for bedifit over collabotation  to only instance where only form steering camera angle is changed
                diff_v1_colab = mAP1_withC - mAP1_new_0 
                diff_v2_colab = mAP2_withC- mAP2_new_0 
                diff_v3_colab = mAP3_withC - mAP3_new_0 
                #print("angles are",diff_v1,diff_v2,diff_v3)
                total_benificial_diff_v1 = diff_v1 
                total_benificial_diff_v2 = diff_v2 
                total_benificial_diff_v3 = diff_v3 
                total_colab =(diff_v1_colab+diff_v2_colab+diff_v3_colab)/100
                cum_total_diff =(total_benificial_diff_v1+total_benificial_diff_v2+1.2*total_benificial_diff_v3)/100
                cummulative_total_colab+=total_colab
                cum_steer_total_diff +=cum_total_diff
                avg_diff_v1 +=diff_v1
                avg_diff_v2 +=diff_v2
                avg_diff_v3 +=diff_v3
                avg_colab_diff_v1+=diff_v1_colab
                avg_colab_diff_v2+=diff_v2_colab
                avg_colab_diff_v3+=diff_v3_colab
              average_steering_benifit = cum_steer_total_diff/self.T
              average_colab_benifit = cummulative_total_colab/self.T
              #additiona_average_steering_benifit  += average_steering_benifit
              avg_diff_v1 = avg_diff_v1/self.T
              avg_diff_v2 = avg_diff_v2/self.T
              avg_diff_v3 = avg_diff_v3/self.T

              avg_colabdiff_v1 = avg_colab_diff_v1/self.T
              avg_colabdiff_v2 = avg_colab_diff_v2/self.T
              avg_colabdiff_v3 = avg_colab_diff_v3/self.T
              self.p1+=avg_colabdiff_v1
              self.p2+=avg_colabdiff_v2
              self.p3+=avg_colabdiff_v3

              if(self.AB == 0 and  self.AC==0 and self.BC==0):
                resipocal_collab_reward = 1.1+average_colab_benifit
              else:
                collab_reward = self.AB+self.AC+self.BC
                resipocal_collab_reward = (1/collab_reward)+average_colab_benifit
              #print(avg_colabdiff_v1,avg_colabdiff_v2,avg_colabdiff_v3)
              #print('(2)',self.p1,self.p2,self.p3)
              #if(avg_diff_v1 >= 0 and avg_diff_v2 >= 0 and avg_diff_v2 >= 0):
              #  reward = self.alpha*average_steering_benifit+ self.beta*resipocal_collab_reward
              #print("##### strat")
              #print(avg_diff_v1 +avg_diff_v2 +avg_diff_v2 )
              #print(avg_diff_v1 ,avg_diff_v2 ,avg_diff_v2 )
              #print("average_steering_benifit,average_colab_benifit",average_steering_benifit,average_colab_benifit,avg_diff_v1,avg_diff_v2,avg_diff_v3)
              #print("#####end")
              ###---------------------------------------------------------------------------------------------------  
              if(average_steering_benifit+average_colab_benifit >= 0):
                reward = self.alpha*average_steering_benifit+ self.beta*resipocal_collab_reward
              elif(average_steering_benifit+average_colab_benifit < 0):
                reward = -1
              if(avg_diff_v1 ==0 and avg_diff_v2 ==0 and avg_diff_v3 ==0):
                reward = 0#self.beta*resipocal_collab_reward
              #if(avg_diff_v1 > 0 and avg_diff_v2 > 0 and avg_diff_v3 > 0 and average_colab_benifit > 0):
              #  reward = 1


              
              #if(average_steering_benifit+average_colab_benifit >= 0):
              #  reward = self.alpha*average_steering_benifit+ self.beta*resipocal_collab_reward
              #  print("(1)","{:.3f}".format(avg_diff_v1),"{:.3f}".format(avg_diff_v2),"{:.3f}".format(avg_diff_v3))#,reward,average_steering_benifit,resipocal_collab_reward)
              #elif(average_steering_benifit+average_colab_benifit < 0):
              #  reward = -1#self.alpha*average_steering_benifit+ self.beta*resipocal_collab_reward#-1
              #  print("(2)","{:.3f}".format(avg_diff_v1),"{:.3f}".format(avg_diff_v2),"{:.3f}".format(avg_diff_v3))
              #if(avg_diff_v1 ==0 and avg_diff_v2 ==0 and avg_diff_v3 ==0):
              #  reward =  0#self.beta*resipocal_collab_reward #0
              #  print("(3)","{:.3f}".format(avg_diff_v1),"{:.3f}".format(avg_diff_v2),"{:.3f}".format(avg_diff_v3))
              #if(avg_diff_v1 > 0 and avg_diff_v2 > 0 and avg_diff_v3 > 0 and average_colab_benifit >0):
              #  reward = 1
              #  print("(4)","{:.3f}".format(avg_diff_v1),"{:.3f}".format(avg_diff_v2),"{:.3f}".format(avg_diff_v3))

              
              #print("reward")
              #print("",reward,average_steering_benifit,resipocal_collab_reward)
              ##------------------------------------------------------------------------------------------------------------------
              #if(avg_diff_v1 +avg_diff_v2 +avg_diff_v3 >= 0 or average_steering_benifit+average_colab_benifit >= 0):
              #  reward = self.alpha*average_steering_benifit+ self.beta*resipocal_collab_reward
              #elif(avg_diff_v1 < 0 or avg_diff_v2 < 0 or avg_diff_v3 < 0):
              #  reward = -1
              #if(avg_diff_v1 ==0 and avg_diff_v2 ==0 and avg_diff_v3 ==0):
              #  reward = 0
              #if(avg_diff_v1 +avg_diff_v2 +avg_diff_v2 >= 0 and average_steering_benifit+average_colab_benifit >= 0):
              #  reward = self.alpha*average_steering_benifit+ self.beta*resipocal_collab_reward
              #elif(avg_diff_v1 < 0 or avg_diff_v2 < 0 or avg_diff_v3 < 0):
              #  reward = -1
              #if(avg_diff_v1 ==0 and avg_diff_v2 ==0 and avg_diff_v3 ==0):
              #  reward = 0#0.05 + self.beta*resipocal_collab_reward
              self.pre_v1_acuracy = avg_diff_v1
              self.pre_v2_acuracy = avg_diff_v2
              self.pre_v3_acuracy = avg_diff_v3
              # resipocal_collab_reward = 0,1,0.5,0.33
              #reward = (diff_v1+diff_v2+diff_v3)/3
              #reward = (mAP1+mAP2+mAP3)/3
              #reward = reward/100
              #print("steering benefit is {}, {}, {} and average_colab_benifit{} and total steering benifit is {}  reward={},  framenumber is {} and action is {}".format(avg_diff_v1,avg_diff_v2,avg_diff_v3,average_colab_benifit,average_steering_benifit,reward ,self.current_state_number,str_exe_action))
              #print("theta1 {} theta2 {} theta3 {} AND exitted  ".format(self.THETA1,self.THETA2,self.THETA3))
              averaged_image = self.preprocess_image(self.c1_view, self.c2_view, self.c3_view)
              #cv2.imwrite("./state_image3.jpg",averaged_image)
              #fps = (time.time() - start_time1)*1000
              #print(f"yyyyyyyFPS: {fps:.2f} ms")

              next_state, reward = averaged_image, reward
              self.current_iteration +=self.T
  
              #print(f"*******************************************FPS: {fps:.2f} ms")
        elif (self.THETA1 < THETA1_LOWER_BOUND_ANGLE or self.THETA2 < THETA2_LOWER_BOUND_ANGLE or  self.THETA3 < THETA3_LOWER_BOUND_ANGLE and self.current_iteration < MAx_ITER_PER_EPISODE): 
           # print("ttt",self.THETA1,self.THETA2,self.THETA3)
            #print("theta1 {} theta2 {} theta3 {} AND exitted  ".format(self.THETA1,self.THETA2,self.THETA3))
            if self.THETA1 < THETA1_LOWER_BOUND_ANGLE:
               self.THETA1 = THETA1_LOWER_BOUND_ANGLE
            if self.THETA2 < THETA2_LOWER_BOUND_ANGLE:
               self.THETA2 = THETA2_LOWER_BOUND_ANGLE
            if self.THETA3 < THETA3_LOWER_BOUND_ANGLE:
               self.THETA3 = THETA3_LOWER_BOUND_ANGLE
            if self.THETA1 > THETA1_UPPER_BOUND_ANGLE:
               self.THETA1 = THETA1_UPPER_BOUND_ANGLE
            if self.THETA2 > THETA2_UPPER_BOUND_ANGLE:
               self.THETA2 = THETA2_UPPER_BOUND_ANGLE
            if self.THETA3 > THETA3_UPPER_BOUND_ANGLE:
               self.THETA3 = THETA3_UPPER_BOUND_ANGLE
            A_x = "A"+str(self.THETA1)
            B_x = "B"+str(self.THETA2)
            C_x = "C"+str(self.THETA3)
            #print("low",A_x,B_x,C_x,value)
            #self.c1_view,_ = self.get_people(view1= 1,view2= 2,view3= 3,value=value,ref=A_x,colab1=B_x,colab2=C_x,colab_action=[self.AB,self.AC,self.BC])
            #self.c2_view,_ = self.get_people(view1= 2,view2= 1,view3= 3,value=value,ref=B_x,colab1=A_x,colab2=C_x,colab_action=[self.AB,self.AC,self.BC])
            #self.c3_view,_ = self.get_people(view1= 3,view2= 1,view3= 2,value=value,ref=C_x,colab1=A_x,colab2=B_x,colab_action=[self.AB,self.AC,self.BC])
            self.c1_view = cam_images[A_x+"_"+str(value)]
            self.c2_view = cam_images[B_x+"_"+str(value)]
            self.c3_view = cam_images[C_x+"_"+str(value)]
            # Compute the average along the last axis
            averaged_image = self.preprocess_image(self.c1_view, self.c2_view, self.c3_view)
            #cv2.imwrite("./state_image2.jpg",averaged_image)
            next_state, reward = averaged_image,-5
            

            self.current_iteration +=self.T
            
            self.done= True 
        elif (self.THETA1 > THETA1_UPPER_BOUND_ANGLE or self.THETA2 > THETA2_UPPER_BOUND_ANGLE or  self.THETA3 > THETA3_UPPER_BOUND_ANGLE and self.current_iteration < MAx_ITER_PER_EPISODE ): 
            #print("ttt",self.THETA1,self.THETA2,self.THETA3)
            #print("theta1 {} theta2 {} theta3 {} AND exitted  ".format(self.THETA1,self.THETA2,self.THETA3))
            if self.THETA1 > THETA1_UPPER_BOUND_ANGLE:
               self.THETA1 = THETA1_UPPER_BOUND_ANGLE
            if self.THETA2 > THETA2_UPPER_BOUND_ANGLE:
               self.THETA2 = THETA2_UPPER_BOUND_ANGLE
            if self.THETA3 > THETA3_UPPER_BOUND_ANGLE:
               self.THETA3 = THETA3_UPPER_BOUND_ANGLE
            if self.THETA1 < THETA1_LOWER_BOUND_ANGLE:
               self.THETA1 = THETA1_LOWER_BOUND_ANGLE
            if self.THETA2 < THETA2_LOWER_BOUND_ANGLE:
               self.THETA2 = THETA2_LOWER_BOUND_ANGLE
            if self.THETA3 < THETA3_LOWER_BOUND_ANGLE:
               self.THETA3 = THETA3_LOWER_BOUND_ANGLE
            A_x = "A"+str(self.THETA1)
            B_x = "B"+str(self.THETA2)
            C_x = "C"+str(self.THETA3)
            #print("high",A_x,B_x,C_x,value)
            #self.c1_view,_ = self.get_people(view1= 1,view2= 2,view3= 3,value=value,ref=A_x,colab1=B_x,colab2=C_x,colab_action=[self.AB,self.AC,self.BC])
            #self.c2_view,_ = self.get_people(view1= 2,view2= 1,view3= 3,value=value,ref=B_x,colab1=A_x,colab2=C_x,colab_action=[self.AB,self.AC,self.BC])
            #self.c3_view,_ = self.get_people(view1= 3,view2= 1,view3= 2,value=value,ref=C_x,colab1=A_x,colab2=B_x,colab_action=[self.AB,self.AC,self.BC])
            self.c1_view = cam_images[A_x+"_"+str(value)]
            self.c2_view = cam_images[B_x+"_"+str(value)]
            self.c3_view = cam_images[C_x+"_"+str(value)]
            averaged_image = self.preprocess_image(self.c1_view, self.c2_view, self.c3_view)
            #cv2.imwrite("./state_image1.jpg",averaged_image)
            next_state, reward = averaged_image,-5
            self.current_iteration += self.T
            self.done= True 

        else:
             next_state, reward = STATE_TERMINAL,-1
             self.done= True 
        info={}
        truncated= False
        
        
        return np.array(next_state), reward,self.done,  truncated, info
    def update_count(self,value, count_dict):
      if value in count_dict:
          count_dict[value] += 1
      else:
          count_dict[value] = 1
    def reset(self,seed=2,THETA1=None, THETA2=None,THETA3=None):
     # print("reset function has been called")
      random_combination = random.choice(combinations)
      sampled_combinations.append(random_combination)
      A_init_angle = random_combination[0]
      B_init_angle = random_combination[1]
      C_init_angle = random_combination[2]
      dic_init_angle_summery[str(random_combination)] += 1
  
      self.current_state_number = 0#random.randint(0, 498)
      if THETA1 is None: THETA1 = A_init_angle#THETA1_LOWER_BOUND_ANGLE
      self.THETA1 = THETA1
      if THETA2 is None: THETA2 = B_init_angle#THETA2_LOWER_BOUND_ANGLE
      self.THETA2 = THETA2
      if THETA3 is None: THETA3 = C_init_angle#THETA3_LOWER_BOUND_ANGLE
      self.THETA3 = THETA3
      observation= self.observe()
      observation = np.array(observation)
      observation=observation.astype(np.uint8)
      
      self.current_iteration=self.current_state_number
      self.update_count(self.current_iteration,dic_episode_lenth)
     # print("reset function has been finished",observation.shape)
      #print(self.THETA1,self.THETA2,self.THETA3)
      self.done= False
      info={}
      return observation,info  # reward, done, info can't be included
    def observe(self):
        IMAGE_RANGE = range(start_image_num, final_image_num)
        A_x = "A"+str(self.THETA1)
        B_x = "B"+str(self.THETA2)
        C_x = "C"+str(self.THETA3)
        
        #value = np.random.choice(IMAGE_RANGE)
        value = self.current_state_number
        self.c1_view,_ = self.get_people(view1= 1,view2= 2,view3= 3,value=value,ref=A_x,colab1=B_x,colab2=C_x,colab_action=[self.AB,self.AC,self.BC])
        self.c2_view,_ = self.get_people(view1= 2,view2= 1,view3= 3,value=value,ref=B_x,colab1=A_x,colab2=C_x,colab_action=[self.AB,self.AC,self.BC])
        self.c3_view,_ = self.get_people(view1= 3,view2= 1,view3= 2,value=value,ref=C_x,colab1=A_x,colab2=B_x,colab_action=[self.AB,self.AC,self.BC])
        #mAP1= self.get_mAP(view =1,image_id=value,det_content=det_content1,ref=A_x,colab1=B_x,colab2=C_x,colab_action=[self.AB,self.AC,self.BC])
        #mAP2= self.get_mAP(view =2,image_id=value,det_content=det_content2,ref=B_x,colab1=A_x,colab2=C_x,colab_action=[self.AB,self.AC,self.BC])
        #mAP3= self.get_mAP(view =3,image_id=value,det_content=det_content3,ref=C_x,colab1=A_x,colab2=B_x,colab_action=[self.AB,self.AC,self.BC])
        #print("to value set",[self.c1c2_area,self.c1c2_people,self.c1c2_occlusion,self.c1c3_area,self.c1c3_people,self.c1c3_occlusion,self.c2c3_area,self.c2c3_people,self.c2c3_occlusion])
        # Compute the average along the last axis
        averaged_image = self.preprocess_image(self.c1_view, self.c2_view, self.c3_view)
        
        #ccc = np.transpose(averaged_image, (1, 2, 0))
        #print("ppppppppppppp",ccc.shape)
        #cv2.imwrite("./state_image2.jpg",ccc)

  
        return averaged_image


    def get_people(self,view1= 1,view2=2,view3=3,value=0,ref='A10',colab1='B10',colab2='C10',colab_action=[]):
      self.en_from_text = 1
      frameID=value
      image_root_path='./YOLOV3/'+dataset_load_name
      foldername_ref  = ref+'/'
      # this condition consider to get results real time by running the NN. if 0 code run the NN. if 1 read from a file.
      if(self.en_from_text == 0):
        # in this method.given two views from two cameras. it should calculate the number of people in the view using yolo.and provide the value
        #foldername_ref  = ref+'/'
        foldername_colab1  = colab1+'/'
        foldername_colab2  = colab2+'/'
        fnal_path_ref = image_root_path +'/' +foldername_ref+"View_00"+str(view1)+'/' 
        fnal_path_colab1 = image_root_path +'/' +foldername_colab1+"View_00"+str(view2)+'/'
        fnal_path_colab2 = image_root_path +'/' +foldername_colab2+"View_00"+str(view3)+'/'
        input_images_ref = load_input_image(fnal_path_ref,value)
        input_images_colab1 = load_input_image(fnal_path_colab1,value)
        input_images_colab2 = load_input_image(fnal_path_colab2,value)
        #pdb.set_trace()
        #print(self.dic_transformation_state.keys())
        #print("pppppppppp",ref+"_"+colab1)
        cori_dic =self.dic_transformation_state[ref+"_"+colab1]
        cori_dic2 =self.dic_transformation_state[ref+"_"+colab2]
        #cori_dic2 =self.dic_transformation_state[ref+"_"+colab2] # as of now i consider only one collaborator
        
        y_pred_colab = ssd_model.predict(input_images_colab1) # input is an image and output conf values 8732,21
        y_pred_colab2 = ssd_model.predict(input_images_colab2) # input is an image and output conf values 8732,21
        y_pred2_ref = ssd_model2.predict(input_images_ref) # input is an image and output conf values 8732,21
        #print("inputtttttttttttttttttttttttttt",input_images2_ref[0,0,:,:])
        y_pred4_ref = ssd_model4.predict(input_images_ref) # input is an image and out put is normal conf,bbobox
        #print(y_pred4_ref[0,1,:])
        y_pred5_colab = ssd_model5.predict(input_images_colab1) # input is an image and out put is normal conf,bbobox
        #y_pred2_ref=colab_process_singleCollaborator(y_pred2_ref,y_pred_colab,cori_dic)
        y_pred2_ref = colab_process_twoCollaborator(y_pred2_ref,y_pred_colab,y_pred_colab2,cori_dic,cori_dic2)
        y_pred3 = ssd_model3.predict(y_pred2_ref)
        final=colab_concatanate_process(y_pred3,y_pred4_ref)
        det_content=[]
        output,det_indices = decode_y2(final,              # this is the ref cam after collabration
                                confidence_thresh=0.25,
                                iou_threshold=0.45,
                                top_k=10,
                                input_coords='centroids',
                                normalize_coords=True,
                                img_height=300,
                                img_width=300)
        #print(output)
        for i in output[0]:
          #print(i,len(output) )
          if(len(i) != 0):
            det_content.append([i[2],i[3],i[4],i[5]])
      else:
        #this part has to be changed accordingly. consider a camera A. at a given moment there are 3 possible views. there are 4 possible instances  for camera A.
        # A without collaboration, A with B, A with C. and A with B and C. so there are 4 possible instances.
        AB=colab_action[0]
        AC=colab_action[1]
        BC=colab_action[2]
        dic_detection_results_index = ""
        index_dic=""
        if(ref[0]=="A"):
          #if(AB==1 and AC==1):
          dic_detection_results_index += "A"
          index_dic += ref
          if(AB==1):
             dic_detection_results_index += "_B"
             index_dic += "_"+colab1
          if(AC==1):  
            dic_detection_results_index += "_C"
            index_dic += "_"+colab2
        elif(ref[0]=="B"):
          dic_detection_results_index += "B"
          index_dic += ref
          if(AB==1):
             dic_detection_results_index += "_A"
             index_dic += "_"+colab1
          if(BC==1):    
            dic_detection_results_index += "_C"
            index_dic += "_"+colab2
        elif(ref[0]=="C"):
          dic_detection_results_index += "C"
          index_dic += ref
          if(AC==1):  
            dic_detection_results_index += "_A"
            index_dic += "_"+colab1
          if(BC==1):    
            dic_detection_results_index += "_B"
            index_dic += "_"+colab2
        temp_dic=dic_detection_results[dic_detection_results_index]
        if(len(str(frameID))==1):
          filename =  "frame_000"+str(frameID)+".jpg"
        elif(len(str(frameID))==2):
         filename = "frame_00"+str(frameID)+".jpg"
        else:
         filename =  "frame_0"+str(frameID)+".jpg"
        index_dic +=  "_View_00"+str(view1)+'_'+filename
        det_content= temp_dic[index_dic] 
      fnal_path_ref = image_root_path_fmap +'/' +foldername_ref+"View_00"+str(view1)+'/' 
      image_out=load_input_fmap_RL(fnal_path_ref,value)
      people = len(det_content)
      image_out=normalize_image_RL(image_out)
      #print('ttttttttttttttttttttttttttttttttttt',det_content)
      return image_out,det_content
    # this code is to read the total area coverage of the camera viewas 
    def get_people_base(self,view1= 1,view2=2,view3=3,value=0,ref='A10',colab1='B10',colab2='C10',colab_action=[]):
      self.en_from_text = 1
      frameID=value
      image_root_path='./YOLOV3/'+dataset_load_name
      foldername_ref  = ref+'/'
      # this condition consider to get results real time by running the NN. if 0 code run the NN. if 1 read from a file.
      if(self.en_from_text == 0):
        # in this method.given two views from two cameras. it should calculate the number of people in the view using yolo.and provide the value
        #foldername_ref  = ref+'/'
        foldername_colab1  = colab1+'/'
        foldername_colab2  = colab2+'/'
        fnal_path_ref = image_root_path +'/' +foldername_ref+"View_00"+str(view1)+'/' 
        fnal_path_colab1 = image_root_path +'/' +foldername_colab1+"View_00"+str(view2)+'/'
        fnal_path_colab2 = image_root_path +'/' +foldername_colab2+"View_00"+str(view3)+'/'
        input_images_ref = load_input_image(fnal_path_ref,value)
        input_images_colab1 = load_input_image(fnal_path_colab1,value)
        input_images_colab2 = load_input_image(fnal_path_colab2,value)
        #pdb.set_trace()
        #print(self.dic_transformation_state.keys())
        #print("pppppppppp",ref+"_"+colab1)
        cori_dic =self.dic_transformation_state[ref+"_"+colab1]
        cori_dic2 =self.dic_transformation_state[ref+"_"+colab2]
        #cori_dic2 =self.dic_transformation_state[ref+"_"+colab2] # as of now i consider only one collaborator
        
        y_pred_colab = ssd_model.predict(input_images_colab1) # input is an image and output conf values 8732,21
        y_pred_colab2 = ssd_model.predict(input_images_colab2) # input is an image and output conf values 8732,21
        y_pred2_ref = ssd_model2.predict(input_images_ref) # input is an image and output conf values 8732,21
        #print("inputtttttttttttttttttttttttttt",input_images2_ref[0,0,:,:])
        y_pred4_ref = ssd_model4.predict(input_images_ref) # input is an image and out put is normal conf,bbobox
        #print(y_pred4_ref[0,1,:])
        y_pred5_colab = ssd_model5.predict(input_images_colab1) # input is an image and out put is normal conf,bbobox
        #y_pred2_ref=colab_process_singleCollaborator(y_pred2_ref,y_pred_colab,cori_dic)
        y_pred2_ref = colab_process_twoCollaborator(y_pred2_ref,y_pred_colab,y_pred_colab2,cori_dic,cori_dic2)
        y_pred3 = ssd_model3.predict(y_pred2_ref)
        final=colab_concatanate_process(y_pred3,y_pred4_ref)
        det_content=[]
        output,det_indices = decode_y2(final,              # this is the ref cam after collabration
                                confidence_thresh=0.25,
                                iou_threshold=0.45,
                                top_k=10,
                                input_coords='centroids',
                                normalize_coords=True,
                                img_height=300,
                                img_width=300)
        #print(output)
        for i in output[0]:
          #print(i,len(output) )
          if(len(i) != 0):
            det_content.append([i[2],i[3],i[4],i[5]])
      else:
        #this part has to be changed accordingly. consider a camera A. at a given moment there are 3 possible views. there are 4 possible instances  for camera A.
        # A without collaboration, A with B, A with C. and A with B and C. so there are 4 possible instances.
        AB=colab_action[0]
        AC=colab_action[1]
        BC=colab_action[2]
        dic_detection_results_index = ""
        index_dic=""
        if(ref[0]=="A"):
          #if(AB==1 and AC==1):
          dic_detection_results_index += "A"
          index_dic += ref
          if(AB==1):
             dic_detection_results_index += "_B"
             index_dic += "_"+colab1
          if(AC==1):  
            dic_detection_results_index += "_C"
            index_dic += "_"+colab2
        elif(ref[0]=="B"):
          dic_detection_results_index += "B"
          index_dic += ref
          if(AB==1):
             dic_detection_results_index += "_A"
             index_dic += "_"+colab1
          if(BC==1):    
            dic_detection_results_index += "_C"
            index_dic += "_"+colab2
        elif(ref[0]=="C"):
          dic_detection_results_index += "C"
          index_dic += ref
          if(AC==1):  
            dic_detection_results_index += "_A"
            index_dic += "_"+colab1
          if(BC==1):    
            dic_detection_results_index += "_B"
            index_dic += "_"+colab2
        temp_dic=dic_detection_results[dic_detection_results_index]
        if(len(str(frameID))==1):
          filename =  "frame_000"+str(frameID)+".jpg"
        elif(len(str(frameID))==2):
         filename = "frame_00"+str(frameID)+".jpg"
        else:
         filename =  "frame_0"+str(frameID)+".jpg"
        index_dic +=  "_View_00"+str(view1)+'_'+filename
        det_content= temp_dic[index_dic] 
      fnal_path_ref = image_root_path +'/' +foldername_ref+"View_00"+str(view1)+'/' 
      image_out=[]#load_input_image(fnal_path_ref,value)
      people = len(det_content)
      image_out=[]#normalize_image(image_out)
      #print('ttttttttttttttttttttttttttttttttttt',det_content)
      return image_out,det_content
    # this code is to read the total area coverage of the camera viewas  
    def get_coverage(self,A='A10',B='B10',C='C10'):
      # in this method.given two views from two cameras. it should calculate the number of people in the view using yolo.and provide the value
      self.en_from_text = 1
      if(self.en_from_text == 0):
         print()
      else:
      #  print("befire",A+B+C)
        index_dic = A+"B"+C[1:]+"C"+B[1:]
       # print("end",index_dic)
        cov_area = self.dic_coverage_accuracy_state[index_dic] 

      return cov_area

    # this module calculate the occulution valur of detected people per frame. therefore i will implement the visualization here in the function.
    def get_occlusion(self,detections):
      occlusion = []
      cumulative_iou=0
      combinations = list(itertools.combinations(detections, 2))
      for people_list in combinations:
          overlap_count = 0
          box1 = people_list[0]
          box2 = people_list[1]
          # calculate overlap using the intersection over union (IoU) metric
          x1 = max(box1[0], box2[0])
          y1 = max(box1[1], box2[1])
          x2 = min(box1[2], box2[2])
          y2 = min(box1[3], box2[3])
          intersection = max(0, x2 - x1) * max(0, y2 - y1)
          area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
          area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
          union = area1 + area2 - intersection
          iou = intersection # / union if union > 0 else 0
          cumulative_iou += iou
          occlusion.append(iou)  # calculate occlusion as a percentage
      if(len(detections) !=0):
        occlusion=(cumulative_iou)/len(detections)
        #print(occlusion)
        occlusion=round(occlusion, 2)
      else:
         occlusion=0
      return occlusion
    # this function rales all coleected occusion data per frame and plot the graph of occulsion vs frame and total occultion of the video segment.


    def plot_occlusion(self,dic_cam1_occlusion,dic_cam2_occlusion,dic_cam3_occlusion,iteration_number,type_im):
     # Plot all three sets of data in the same graph
      plt2.plot(list(dic_cam1_occlusion.keys()), list(dic_cam1_occlusion.values()), label="Camera 1")
      plt2.plot(list(dic_cam2_occlusion.keys()), list(dic_cam2_occlusion.values()), label="Camera 2")
      plt2.plot(list(dic_cam3_occlusion.keys()), list(dic_cam3_occlusion.values()), label="Camera 3")
      plt2.xlabel("X-axis")
      plt2.ylabel("Y-axis")
      plt2.title(type_im)
      plt2.legend()

      # Calculate total occlusion values for each camera
      total_cam1_occlusion = sum(dic_cam1_occlusion.values())
      total_cam2_occlusion = sum(dic_cam2_occlusion.values())
      total_cam3_occlusion = sum(dic_cam3_occlusion.values())
      average_occulusion= (total_cam1_occlusion+total_cam2_occlusion+total_cam3_occlusion)
      # Print the total occlusion values for each camera
      print("Total occlusion for Camera 1:", total_cam1_occlusion)
      print("Total occlusion for Camera 2:", total_cam2_occlusion)
      print("Total occlusion for Camera 3:", total_cam3_occlusion)
      print("Total camera occulation is:", average_occulusion)
      # Save the plot as an image file
    #  plt2.savefig(str(type_im)+".png")
      plt2.clf()
      
      # Log the statements and print the variables
    #  logger.info("executing type of process is %s", type_im)
    #  logger.info("Total occlusion for Camera 1: %s", total_cam1_occlusion)
    #  logger.info("Total occlusion for Camera 2: %s", total_cam2_occlusion)
    #  logger.info("Total occlusion for Camera 3: %s", total_cam3_occlusion)
    #  logger.info("Total camera occlusion is: %s", average_occulusion)
      # Show the plot
      #plt.show()
      return 0
    def plot_able_variation(self,dic_cam1_occlusion,dic_cam2_occlusion,dic_cam3_occlusion,iteration_number,type_im):
     # Plot all three sets of data in the same graph
      plt2.plot(list(dic_cam1_occlusion.keys()), list(dic_cam1_occlusion.values()), label="Camera 1")
      plt2.plot(list(dic_cam2_occlusion.keys()), list(dic_cam2_occlusion.values()), label="Camera 2")
      plt2.plot(list(dic_cam3_occlusion.keys()), list(dic_cam3_occlusion.values()), label="Camera 3")
      plt2.xlabel("X-axis")
      plt2.ylabel("Y-axis")
      plt2.title( "Angle Variation per frame")
      plt2.legend()
      # Save the plot as an image file
      plt2.savefig(str(type_im)+".png")
      plt2.clf()

      # Show the plot
      #plt.show()
      return 0

    def plot_able_variation2(self,A,B,C):
     # Plot all three sets of data in the same graph
      x=[i for i in range(500)]
      AB=[i[0] for i in A]
      AB_overlap=[i[1] for i in A]
      AC=[2+i[0] for i in B]
      AC_overlap=[2+i[1] for i in B]
      BC=[5+i[0] for i in C]
      BC_overlap=[5+i[1] for i in C]

      plt2.plot(x, AB, label="AB")
      plt2.plot(x, AC, label="AC")
      plt2.plot(x, BC , label="BC")
      plt2.plot(x, AB_overlap, label="AB_overlap 1")
      plt2.plot(x, AC_overlap, label="AC_overlap 2")
      plt2.plot(x, BC_overlap , label="BC_overlap 3")
      plt2.xlabel("X-axis")
      plt2.ylabel("Y-axis")
      plt2.title( "Angle Variation per frame")
      plt2.legend()
      # Save the plot as an image file
      plt2.savefig("./collab.png")
      plt2.clf()

      # Show the plot
      #plt.show()
      return 0

    def get_mAP(self,view= 1,image_id=0,det_content=[],ref="A10",colab1="B10",colab2="C10",colab_action=[]):
      mAP=10
      value=image_id
      frameID=value
      if(len(str(value))==1):
          imageName =  "frame_000"+str(value)+".jpg"
      elif(len(str(value))==2):
          imageName = "frame_00"+str(value)+".jpg"
      else:
          imageName = "frame_0"+str(value)+".jpg"
      self.en_from_text = 1
      if(self.en_from_text == 0):
         final_gt_path='./YOLOV3/'+dataset_load_name+'/'+A+"/"
         initialize_priors(view,final_gt_path)
         calc_accuracy(view,image_id,det_content,iou_tre=float(0.5),A=A)
         Precision,Recall,F1Score,TP,FP,FN,GT=get_Accuracy_Metrics()
      else:
        #index_dic = A + "_View_00"+str(view)+'_'+imageName
        #this part has to be changed accordingly. consider a camera A. at a given moment there are 3 possible views. there are 4 possible instances  for camera A.
        # A without collaboration, A with B, A with C. and A with B and C. so there are 4 possible instances.
        AB=colab_action[0]
        AC=colab_action[1]
        BC=colab_action[2]
        dic_detection_results_index = ""
        index_dic=""
        if(ref[0]=="A"):
          #if(AB==1 and AC==1):
          dic_detection_results_index += "A"
          index_dic += ref
          if(AB==1):
             dic_detection_results_index += "_B"
             index_dic += "_"+colab1
          if(AC==1):  
            dic_detection_results_index += "_C"
            index_dic += "_"+colab2
        elif(ref[0]=="B"):
          dic_detection_results_index += "B"
          index_dic += ref
          if(AB==1):
             dic_detection_results_index += "_A"
             index_dic += "_"+colab1
          if(BC==1):    
            dic_detection_results_index += "_C"
            index_dic += "_"+colab2
        elif(ref[0]=="C"):
          dic_detection_results_index += "C"
          index_dic += ref
          if(AC==1):  
            dic_detection_results_index += "_A"
            index_dic += "_"+colab1
          if(BC==1):    
            dic_detection_results_index += "_B"
            index_dic += "_"+colab2
        temp_dic=dic_mAP_results[dic_detection_results_index]
        if(len(str(frameID))==1):
          filename =  "frame_000"+str(frameID)+".jpg"
        elif(len(str(frameID))==2):
         filename = "frame_00"+str(frameID)+".jpg"
        else:
         filename =  "frame_0"+str(frameID)+".jpg"
        index_dic +=  "_View_00"+str(view)+'_'+filename
        F1Score= temp_dic[index_dic]
      if(ENABLE_PRINT):
           print("Precision,Recall,F1Score,TP,FP,FN,GT",Precision,Recall,F1Score,TP,FP,FN,GT)
      return F1Score 
      #start game


    def get_TPFPFN(self,view= 1,image_id=0,det_content=[],theta1=0,theta2=0,theta3=0,ref="A10",colab1="B10",colab2="C10",colab_action=[]):
      mAP=10
      #foldername  = 'A'+str(theta1)+'B'+str(theta2)+'C'+str(theta3)+'/'
      #final_gt_path='./YOLOV3/final_sim_images/'+A+"/"
      #initialize_priors(view,final_gt_path)
      #calc_accuracy(view,image_id,det_content,iou_tre=float(0.5),A=A)
      #Precision,Recall,F1Score,TP,FP,FN,GT=get_Accuracy_Metrics()
      value=image_id
      image_root_path='./YOLOV3/'+dataset_load_name
      frameID=value
      if(len(str(value))==1):
          imageName =  "frame_000"+str(value)+".jpg"
      elif(len(str(value))==2):
          imageName = "frame_00"+str(value)+".jpg"
      else:
          imageName = "frame_0"+str(value)+".jpg"
      self.en_from_text = 1
      if(self.en_from_text == 0):
         final_gt_path='./YOLOV3/'+dataset_load_name+'/'+A+"/"
         initialize_priors(view,final_gt_path)
         calc_accuracy(view,image_id,det_content,iou_tre=float(0.5),A=A)
         Precision,Recall,F1Score,TP,FP,FN,GT=get_Accuracy_Metrics()
      else:
        #this part has to be changed accordingly. consider a camera A. at a given moment there are 3 possible views. there are 4 possible instances  for camera A.
        # A without collaboration, A with B, A with C. and A with B and C. so there are 4 possible instances.
        AB=colab_action[0]
        AC=colab_action[1]
        BC=colab_action[2]
        dic_detection_results_index = ""
        index_dic=""
        if(ref[0]=="A"):
          #if(AB==1 and AC==1):
          dic_detection_results_index += "A"
          index_dic += ref
          if(AB==1):
             dic_detection_results_index += "_B"
             index_dic += "_"+colab1
          if(AC==1):  
            dic_detection_results_index += "_C"
            index_dic += "_"+colab2
        elif(ref[0]=="B"):
          dic_detection_results_index += "B"
          index_dic += ref
          if(AB==1):
             dic_detection_results_index += "_A"
             index_dic += "_"+colab1
          if(BC==1):    
            dic_detection_results_index += "_C"
            index_dic += "_"+colab2
        elif(ref[0]=="C"):
          dic_detection_results_index += "C"
          index_dic += ref
          if(AC==1):  
            dic_detection_results_index += "_A"
            index_dic += "_"+colab1
          if(BC==1):    
            dic_detection_results_index += "_B"
            index_dic += "_"+colab2
        temp_dic=dic_TPFPFN_results[dic_detection_results_index]
        if(len(str(frameID))==1):
          filename =  "frame_000"+str(frameID)+".jpg"
        elif(len(str(frameID))==2):
         filename = "frame_00"+str(frameID)+".jpg"
        else:
         filename =  "frame_0"+str(frameID)+".jpg"
        index_dic +=  "_View_00"+str(view)+'_'+filename
        TP,FP,FN= temp_dic[index_dic] 
      foldername_ref  = ref+'/'
      fnal_path_ref = image_root_path +'/' +foldername_ref+"View_00"+str(view)+'/' 
      #image_out=load_input_image(fnal_path_ref,value)
      #people = len(det_content)
      #image_out=normalize_image(image_out)
      if(ENABLE_PRINT):
        print("Precision,Recall,F1Score,TP,FP,FN,GT",Precision,Recall,F1Score,TP,FP,FN,GT)
      return TP,FP,FN
 
    def plot_actions(self,dic_action,iteration_number,type_im):
        # Plot all three sets of data in the same graph
         #plt.bar(x, y)
         plt.bar(list(dic_action.keys()), list(dic_action.values()), label="Actions")
         plt.xlabel("X-axis")
         plt.ylabel("Y-axis")
         plt.title("action_explored")
         plt.legend()
         plt.xticks(list(dic_action.keys()))
         plt.xticks(rotation=45)
         
         # Save the plot as an image file
         plt.savefig("./plots/action_explored_"+type_im+"_"+str(iteration_number)+".png")
         plt.clf()
         # Show the plot
         #plt.show()
         return 0
    def plot_accuracy_per_frame(self,A,B,C,A1,B1,C1,iteration_number):
         # Create a figure with three subplots arranged horizontally
        fig, axs = plt3.subplots(1, 3, figsize=(15, 5))
        # Plot data for Camera 1

        axs[0].plot(list(A.keys()), list(A.values()), label="C1 baseline Fscore")
        axs[0].plot(list(A1.keys()), list(A1.values()), label="C1 DQN Fscore")
     
        axs[0].set_xlabel("X-axis")
        axs[0].set_ylabel("Y-axis")
        axs[0].set_title("Camera 1 Fscore")
        axs[0].legend()
        # Plot data for Camera 2
        axs[1].plot(list(B.keys()), list(B.values()), label="C2 baseline Fscore")
        axs[1].plot(list(B1.keys()), list(B1.values()), label="C2 DQN Fscore")
   
        axs[1].set_xlabel("X-axis")
        axs[1].set_ylabel("Y-axis")
        axs[1].set_title("Camera 2 Fscore")
        axs[1].legend()
        # Plot data for Camera 3
        axs[2].plot(list(C.keys()), list(C.values()), label="C3 baseline Fscore")
        axs[2].plot(list(C.keys()), list(C1.values()), label="C3 DQN Fscore")
 
        axs[2].set_xlabel("X-axis")
        axs[2].set_ylabel("Y-axis")
        axs[2].set_title("Camera 3 Fscore")
        axs[2].legend()
        # Save the plot as an image file
        plt3.savefig("./plots/accuracy_per_frame_"+str(iteration_number)+".png")
        plt3.clf()
        # Show the plot
        #plt.show()
        return 0
    def plot_avg_accuracy_per_frame_with_colab(self,A,B,C,A1,B1,C1,A_colab,B_colab,C_colab,iteration_number):
        # Create a figure with three subplots arranged horizontally
       # plt4.figure(figsize=(70, 20)) 
        plt4.xlabel("X-axis")
        plt4.ylabel("Y-axis")
        plt4.title("avg Fscore per frame")
        plt4.legend()
        plt4.xticks(list(A.keys()))
        plt4.xticks(rotation=90)
        # Save the plot as an image file
        list1=list(A.values())
        list2=list(B.values())
        list3=list(C.values())

        cam_a = list(A_colab.values())

        list4=list(A1.values())
        list5=list(B1.values())
        list6=list(C1.values())
        # Plot data for Camera 1
        avg_list_base=[]
        new_dic={}
        for i in range(0,len(list(A.keys()))):
          avg = (list1[i][0]+list2[i][0]+list3[i][0])/3
          avg_list_base.append(avg)
        avg_list_dqn=[]
        for i in range(0,len(list(A.keys()))):
          avg = (list4[i][0]+list5[i][0]+list6[i][0])/3
          avg_list_dqn.append(avg)
          new_dic[i]=avg
        camA_colab_list=[]
        for i in range(0,len(list(A_colab.keys()))):
          avg = (cam_a[i])*10
          camA_colab_list.append(avg)
        
        #plt4.plot(list(A.keys()), avg_list_base, label="baseline Fscore")
        #print(new_dic)
        bar_width=0.4
        index=range(len(list(A.keys())))
        #plt4.bar(index, avg_list_base, width=bar_width, label='Accuracy baseline', alpha=0.8)
        #plt4.bar(index, avg_list_dqn, width=bar_width, label='Accuracy dqn', alpha=0.8)
        plt4.plot(list(A.keys()), avg_list_base, label="base Fscore")
        plt4.plot(list(A.keys()), avg_list_dqn, label="DQN Fscore")
        plt4.plot(list(A.keys()), camA_colab_list, label="Cam A colab behavior")
        plt4.margins(x=0)
        plt4.xticks(index, list(A.keys()),fontsize=20)
        plt4.legend()
        plt4.savefig("./plots/avg_fscore_perFrame_"+str(iteration_number)+".png")
        plt4.clf()
        return 0 
    def plot_avg_accuracy_per_frame_with_colab_pure(self,A,B,C,A1,B1,C1,A_colab,B_colab,C_colab,iteration_number):
        # Create a figure with three subplots arranged horizontally
       # plt4.figure(figsize=(70, 20)) 
        plt4.xlabel("X-axis")
        plt4.ylabel("Y-axis")
        plt4.title("avg Fscore per frame")
        plt4.legend()
        plt4.xticks(list(A.keys()))
        plt4.xticks(rotation=90)
        # Save the plot as an image file
        list1=list(A.values())
        list2=list(B.values())
        list3=list(C.values())

        cam_a = list(A_colab.values())
        cam_b = list(B_colab.values())
        cam_c = list(C_colab.values())
        print(cam_a)
        print(cam_b)
        print(cam_c)
        list4=list(A1.values())
        list5=list(B1.values())
        list6=list(C1.values())
        # Plot data for Camera 1
        avg_list_base=[]
        new_dic={}
        for i in range(0,len(list(A.keys()))):
          avg = (list1[i][0]+list2[i][0]+list3[i][0])/3
          avg_list_base.append(avg)
        avg_list_dqn=[]
        for i in range(0,len(list(A.keys()))):
          avg = (list4[i][0]+list5[i][0]+list6[i][0])/3
          avg_list_dqn.append(avg)
          new_dic[i]=avg
        camA_colab_list=[]
        camB_colab_list=[]
        camC_colab_list=[]
        for i in range(0,len(list(A_colab.keys()))):
          avg = (cam_a[i])*10+100
          camA_colab_list.append(avg)
        for i in range(0,len(list(B_colab.keys()))):
          avg = (cam_b[i])*10+130
          camB_colab_list.append(avg)
        for i in range(0,len(list(C_colab.keys()))):
          avg = (cam_c[i])*10+160
          camC_colab_list.append(avg)
        
        #plt4.plot(list(A.keys()), avg_list_base, label="baseline Fscore")
        #print(new_dic)
        bar_width=0.4
        index=range(len(list(A.keys())))
        #plt4.bar(index, avg_list_base, width=bar_width, label='Accuracy baseline', alpha=0.8)
        #plt4.bar(index, avg_list_dqn, width=bar_width, label='Accuracy dqn', alpha=0.8)
        plt4.plot(list(A.keys()), avg_list_base, label="two collaborators Fscore")
        plt4.plot(list(A.keys()), avg_list_dqn, label="DQN Fscore")
        plt4.plot(list(A.keys()), camA_colab_list, label="Cam A colab behavior")
        plt4.plot(list(A.keys()), camB_colab_list, label="Cam B colab behavior")
        plt4.plot(list(A.keys()), camC_colab_list, label="Cam C colab behavior")
        plt4.margins(x=0)
        plt4.xticks(index, list(A.keys()),fontsize=20)
        plt4.legend()
        plt4.savefig("./plots/pure_avg_fscore_perFrame_"+str(iteration_number)+".png")
        plt4.clf()
        return 0 
    def plot_avg_accuracy_per_frame_with_colab_three(self,A,B,C,A1,B1,C1,A_colab,B_colab,C_colab,iteration_number):
        # Create a figure with three subplots arranged horizontally
       # plt4.figure(figsize=(70, 20)) 
        plt4.xlabel("X-axis")
        plt4.ylabel("Y-axis")
        plt4.title("avg Fscore per frame")
        plt4.legend()
        plt4.xticks(list(A.keys()))
        plt4.xticks(rotation=90)
        # Save the plot as an image file
        list1=list(A.values())
        list2=list(B.values())
        list3=list(C.values())

        cam_a = list(A_colab.values())
        cam_b = list(B_colab.values())
        cam_c = list(C_colab.values())
        #print(cam_a)
        #print(cam_b)
        #print(cam_c)
        list4=list(A1.values())
        list5=list(B1.values())
        list6=list(C1.values())
        # Plot data for Camera 1
        avg_list_base=[]
        new_dic={}
        dd=0
        for i in range(0,len(list(A.keys()))):
          avg = (list1[i][0]+list2[i][0]+list3[i][0])/3
          avg_list_base.append(avg)
        avg_list_dqn=[]
        for i in range(0,len(list(A.keys()))):
          avg = (list4[i][0]+list5[i][0]+list6[i][0])/3
          avg_list_dqn.append(avg)
          dd+=avg
          new_dic[i]=avg
        new_dic_3 =[]
        for i in range(0,len(list(A.keys()))):
          avg = (cam_a[i][0]+cam_b[i][0]+cam_c[i][0])/3
          new_dic_3.append(avg)
        print("two collaborator accuracy is",dd/500)

        
        #plt4.plot(list(A.keys()), avg_list_base, label="baseline Fscore")
        #print(new_dic)
        bar_width=0.4
        index=range(len(list(A.keys())))
        #plt4.bar(index, avg_list_base, width=bar_width, label='Accuracy baseline', alpha=0.8)
        #plt4.bar(index, avg_list_dqn, width=bar_width, label='Accuracy dqn', alpha=0.8)
        plt4.plot(list(A.keys()), avg_list_base, label="beaseline Fscore")
        plt4.plot(list(A.keys()), avg_list_dqn, label="two collaborators Fscore")
        plt4.plot(list(A.keys()), new_dic_3, label="DQN colab selection Fscore")

        plt4.margins(x=0)
        plt4.xticks(index, list(A.keys()),fontsize=20)
        plt4.legend()
        plt4.savefig("./plots/three_avg_fscore_perFrame_"+str(iteration_number)+".png")
        plt4.clf()
        return 0 
    def plot_avg_accuracy_per_frame(self,A,B,C,A1,B1,C1,iteration_number):
        # Create a figure with three subplots arranged horizontally
       # plt4.figure(figsize=(70, 20)) 
        plt4.xlabel("X-axis")
        plt4.ylabel("Y-axis")
        plt4.title("avg Fscore per frame")
        plt4.legend()
        plt4.xticks(list(A.keys()))
        plt4.xticks(rotation=90)
        # Save the plot as an image file
        list1=list(A.values())
        list2=list(B.values())
        list3=list(C.values())

        list4=list(A1.values())
        list5=list(B1.values())
        list6=list(C1.values())
        # Plot data for Camera 1
        avg_list_base=[]
        new_dic={}
        for i in range(0,len(list(A.keys()))):
          avg = (list1[i][0]+list2[i][0]+list3[i][0])/3
          avg_list_base.append(avg)
        avg_list_dqn=[]
        for i in range(0,len(list(A.keys()))):
          avg = (list4[i][0]+list5[i][0]+list6[i][0])/3
          avg_list_dqn.append(avg)
          new_dic[i]=avg
        #plt4.plot(list(A.keys()), avg_list_base, label="baseline Fscore")
        #print(new_dic)
        bar_width=0.4
        index=range(len(list(A.keys())))
        #plt4.bar(index, avg_list_base, width=bar_width, label='Accuracy baseline', alpha=0.8)
        #plt4.bar(index, avg_list_dqn, width=bar_width, label='Accuracy dqn', alpha=0.8)
        plt4.plot(list(A.keys()), avg_list_base, label="base Fscore")
        plt4.plot(list(A.keys()), avg_list_dqn, label="DQN Fscore")
        plt4.margins(x=0)
        plt4.xticks(index, list(A.keys()),fontsize=20)
        plt4.legend()
        plt4.savefig("./plots/avg_fscore_perFrame_"+str(iteration_number)+".png")
        plt4.clf()
        return 0 
 #   self.humman, self.computer = True, False
    def startGame(self, model,iteration_number):
      #dic_test_summery_action={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0,19:0,20:0,21:0,22:0,23:0,24:0,25:0,26:0}
      dic_test_summery_action={0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0, 39: 0, 40: 0, 41: 0, 42: 0, 43: 0, 44: 0, 45: 0, 46: 0, 47: 0, 48: 0, 49: 0, 50: 0, 51: 0, 52: 0, 53: 0, 54: 0, 55: 0, 56: 0, 57: 0, 58: 0, 59: 0, 60: 0, 61: 0, 62: 0, 63: 0, 64: 0, 65: 0, 66: 0, 67: 0, 68: 0, 69: 0, 70: 0, 71: 0, 72: 0, 73: 0, 74: 0, 75: 0, 76: 0, 77: 0, 78: 0, 79: 0, 80: 0, 81: 0, 82: 0, 83: 0, 84: 0, 85: 0, 86: 0, 87: 0, 88: 0, 89: 0, 90: 0, 91: 0, 92: 0, 93: 0, 94: 0, 95: 0, 96: 0, 97: 0, 98: 0, 99: 0, 100: 0, 101: 0, 102: 0, 103: 0, 104: 0, 105: 0, 106: 0, 107: 0, 108: 0, 109: 0, 110: 0, 111: 0, 112: 0, 113: 0, 114: 0, 115: 0, 116: 0, 117: 0, 118: 0, 119: 0, 120: 0, 121: 0, 122: 0, 123: 0, 124: 0, 125: 0, 126: 0, 127: 0, 128: 0, 129: 0, 130: 0, 131: 0, 132: 0, 133: 0, 134: 0, 135: 0, 136: 0, 137: 0, 138: 0, 139: 0, 140: 0, 141: 0, 142: 0, 143: 0, 144: 0, 145: 0, 146: 0, 147: 0, 148: 0, 149: 0, 150: 0, 151: 0, 152: 0, 153: 0, 154: 0, 155: 0, 156: 0, 157: 0, 158: 0, 159: 0, 160: 0, 161: 0, 162: 0, 163: 0, 164: 0, 165: 0, 166: 0, 167: 0, 168: 0, 169: 0, 170: 0, 171: 0, 172: 0, 173: 0, 174: 0, 175: 0, 176: 0, 177: 0, 178: 0, 179: 0, 180: 0, 181: 0, 182: 0, 183: 0, 184: 0, 185: 0, 186: 0, 187: 0, 188: 0, 189: 0, 190: 0, 191: 0, 192: 0, 193: 0, 194: 0, 195: 0, 196: 0, 197: 0, 198: 0, 199: 0, 200: 0, 201: 0, 202: 0, 203: 0, 204: 0, 205: 0, 206: 0, 207: 0, 208: 0, 209: 0, 210: 0, 211: 0, 212: 0, 213: 0, 214: 0, 215: 0}
      if (1): #if AI     
        self.reset()
        done = False
        # this is for the baseline accuracy calculation

        init_angle_THETA1 = 0
        init_angle_THETA2 = 10
        init_angle_THETA3 = 0
        print("loading inital angles are",init_angle_THETA1,init_angle_THETA2,init_angle_THETA3)
        THETA1_LOWER_BOUND_ANGLE = 0
        THETA2_LOWER_BOUND_ANGLE = 0
        THETA3_LOWER_BOUND_ANGLE = 0
        cul_TP1,cul_FP1,cul_FN1 =0,0,0
        cul_TP2,cul_FP2,cul_FN2 =0,0,0
        cul_TP3,cul_FP3,cul_FN3 =0,0,0
        RL_cul_TP1,RL_cul_FP1,RL_cul_FN1 =0,0,0
        RL_cul_TP2,RL_cul_FP2,RL_cul_FN2 =0,0,0
        RL_cul_TP3,RL_cul_FP3,RL_cul_FN3 =0,0,0

        RL_cul_TP1_all,RL_cul_FP1_all,RL_cul_FN1_all =0,0,0
        RL_cul_TP2_all,RL_cul_FP2_all,RL_cul_FN2_all =0,0,0
        RL_cul_TP3_all,RL_cul_FP3_all,RL_cul_FN3_all =0,0,0

        RL_THETA1= init_angle_THETA1
        RL_THETA2= init_angle_THETA2
        RL_THETA3= init_angle_THETA3

        dic_cam1_base_occlusion={}
        dic_cam2_base_occlusion={}
        dic_cam3_base_occlusion={}
        dic_cam1_DQN_occlusion={}
        dic_cam2_DQN_occlusion={}
        dic_cam3_DQN_occlusion={}
        dic_cam1_angle_variation={}
        dic_cam2_angle_variation={}
        dic_cam3_angle_variation={}
        dic_cam1_baseline_accuracy={}
        dic_cam2_baseline_accuracy={}
        dic_cam3_baseline_accuracy={}
        dic_cam1_DQN_accuracy={}
        dic_cam2_DQN_accuracy={}
        dic_cam3_DQN_accuracy={}
        dic_camA_colab_behavior={}
        dic_camB_colab_behavior={}
        dic_camC_colab_behavior={}
        
        dic_cam1_colab_accuracy={}
        dic_cam2_colab_accuracy={}
        dic_cam3_colab_accuracy={}

        dic_outofbound={}
        set1_temp_c1_tp=0
        set1_temp_c1_fp=0
        set1_temp_c1_fn=0
        set1_temp_c2_tp=0
        set1_temp_c2_fp=0
        set1_temp_c2_fn=0
        set1_temp_c3_tp=0
        set1_temp_c3_fp=0
        set1_temp_c3_fn=0

        set2_temp_c1_tp=0
        set2_temp_c1_fp=0
        set2_temp_c1_fn=0
        set2_temp_c2_tp=0
        set2_temp_c2_fp=0
        set2_temp_c2_fn=0
        set2_temp_c3_tp=0
        set2_temp_c3_fp=0
        set2_temp_c3_fn=0

        
        for video_frame in range (0,500):
          
          base_line_THETA1= init_angle_THETA1
          base_line_THETA2= init_angle_THETA2
          base_line_THETA3= init_angle_THETA3
          A_x = "A"+str(base_line_THETA1)
          B_x = "B"+str(base_line_THETA2)
          C_x = "C"+str(base_line_THETA3)
          print("loading pure anglea are",A_x,B_x,C_x)
          yy1,det_content1 = self.get_people(view1= 1,view2= 2,view3= 3,value=video_frame,ref=A_x,colab1=B_x,colab2=C_x,colab_action=[0,0,0])
          yy2,det_content2 = self.get_people(view1= 2,view2= 1,view3= 3,value=video_frame,ref=B_x,colab1=A_x,colab2=C_x,colab_action=[0,0,0])
          yy3,det_content3 = self.get_people(view1= 3,view2= 1,view3= 2,value=video_frame,ref=C_x,colab1=A_x,colab2=B_x,colab_action=[0,0,0])

          base_covering_area = self.get_coverage(A_x,B_x,C_x)

          TP1,FP1,FN1= self.get_TPFPFN(view =1,image_id=video_frame,det_content=det_content1,theta1 = base_line_THETA1 ,theta2 = base_line_THETA2,theta3 = base_line_THETA3,ref=A_x,colab1=B_x,colab2=C_x,colab_action=[0,0,0])
          TP2,FP2,FN2= self.get_TPFPFN(view =2,image_id=video_frame,det_content=det_content2,theta1 = base_line_THETA1 ,theta2 = base_line_THETA2,theta3 = base_line_THETA3,ref=B_x,colab1=A_x,colab2=C_x,colab_action=[0,0,0])
          TP3,FP3,FN3= self.get_TPFPFN(view =3,image_id=video_frame,det_content=det_content3,theta1 = base_line_THETA1 ,theta2 = base_line_THETA2,theta3 = base_line_THETA3,ref=C_x,colab1=A_x,colab2=B_x,colab_action=[0,0,0])
          cul_TP1+= TP1
          cul_FP1+= FP1
          cul_FN1+= FN1
          cul_TP2+= TP2
          cul_FP2+= FP2
          cul_FN2+= FN2
          cul_TP3+= TP3
          cul_FP3+= FP3
          cul_FN3+= FN3
          if(TP1+FP1 != 0):
            current_c1_baseline_pricision = float(TP1)/(TP1+FP1)*100.0
          else:
            current_c1_baseline_pricision = 0

          if(TP2+FP2 != 0):
            current_c2_baseline_pricision = float(TP2)/(TP2+FP2)*100.0
          else:
            current_c2_baseline_pricision = 0

          if(TP3+FP3 != 0):
            current_c3_baseline_pricision = float(TP3)/(TP3+FP3)*100.0
          else:
            current_c3_baseline_pricision = 0

          if(TP1+FN1 != 0):
            current_c1_baseline_recall = float(TP1)/(TP1+FN1)*100.0
          else:
            current_c1_baseline_recall = 0

          if(TP2+FN2 != 0):
            current_c2_baseline_recall = float(TP2)/(TP2+FN2)*100.0
          else:
            current_c2_baseline_recall = 0

          if(TP3+FN3 != 0):
            current_c3_baseline_recall = float(TP3)/(TP3+FN3)*100.0
          else:
            current_c3_baseline_recall = 0


          if(current_c1_baseline_pricision+current_c1_baseline_recall):
            current_c1_baseline_F1Score = 2*current_c1_baseline_pricision*current_c1_baseline_recall/(current_c1_baseline_pricision+current_c1_baseline_recall)
          else:
            current_c1_baseline_F1Score = 0

          if(current_c2_baseline_pricision+current_c2_baseline_recall):
            current_c2_baseline_F1Score = 2*current_c2_baseline_pricision*current_c2_baseline_recall/(current_c2_baseline_pricision+current_c2_baseline_recall)
          else:
            current_c2_baseline_F1Score = 0

          if(current_c3_baseline_pricision+current_c3_baseline_recall):
            current_c3_baseline_F1Score = 2*current_c3_baseline_pricision*current_c3_baseline_recall/(current_c3_baseline_pricision+current_c3_baseline_recall)
          else:
            current_c3_baseline_F1Score = 0

          occlusion1_base_cam1 = self.get_occlusion(det_content1)
          occlusion2_base_cam2 = self.get_occlusion(det_content2)
          occlusion3_base_cam3 = self.get_occlusion(det_content3)
          dic_cam1_base_occlusion[video_frame] = occlusion1_base_cam1
          dic_cam2_base_occlusion[video_frame] = occlusion2_base_cam2
          dic_cam3_base_occlusion[video_frame] = occlusion3_base_cam3

          # below code is for dynamic angle configurations
          #method 1 :
          # in this method i chnage the theta values after 5 frames  ACTION_THETA1_INC,ACTION_THETA1_DEC,ACTION_THETA2_INC,ACTION_THETA2_DEC,ACTION_THETA3_INC,ACTION_THETA3_DEC,DO_NOTHING
          #action_list=[ACTION_THETA1_INC,ACTION_THETA1_DEC,ACTION_THETA2_INC,ACTION_THETA2_DEC,ACTION_THETA3_INC,ACTION_THETA3_DEC,DO_NOTHING]
          if RL_THETA1 < THETA1_LOWER_BOUND_ANGLE:
             dic_outofbound[video_frame] = "braked"
             break
          elif RL_THETA1 > THETA1_UPPER_BOUND_ANGLE:
             dic_outofbound[video_frame] = "braked"
             break
          if RL_THETA2 < THETA2_LOWER_BOUND_ANGLE:
             dic_outofbound[video_frame] = "braked"
             break
          elif RL_THETA2 > THETA2_UPPER_BOUND_ANGLE:
             dic_outofbound[video_frame] = "braked"
             break
          if RL_THETA3 < THETA3_LOWER_BOUND_ANGLE:
             dic_outofbound[video_frame] = "braked"
             break
          elif RL_THETA3 > THETA3_UPPER_BOUND_ANGLE:
             dic_outofbound[video_frame] = "braked"
             break
          A_x = "A"+str(RL_THETA1)
          B_x = "B"+str(RL_THETA2)
          C_x = "C"+str(RL_THETA3)
          
          
          xx1,det_content1 = self.get_people(view1= 1,view2= 2,view3= 3,value=video_frame,ref=A_x,colab1=B_x,colab2=C_x,colab_action=[self.AB,self.AC,self.BC])
          xx2,det_content2 = self.get_people(view1= 2,view2= 1,view3= 3,value=video_frame,ref=B_x,colab1=A_x,colab2=C_x,colab_action=[self.AB,self.AC,self.BC])
          xx3,det_content3 = self.get_people(view1= 3,view2= 1,view3= 2,value=video_frame,ref=C_x,colab1=A_x,colab2=B_x,colab_action=[self.AB,self.AC,self.BC])
          dqn_covering_area = self.get_coverage(A_x,B_x,C_x)
          TP1,FP1,FN1= self.get_TPFPFN(view =1,image_id=video_frame,det_content=det_content1,theta1 = RL_THETA1 ,theta2 = RL_THETA2,theta3 = RL_THETA3,ref=A_x,colab1=B_x,colab2=C_x,colab_action=[self.AB,self.AC,self.BC])
          TP2,FP2,FN2= self.get_TPFPFN(view =2,image_id=video_frame,det_content=det_content2,theta1 = RL_THETA1 ,theta2 = RL_THETA2,theta3 = RL_THETA3,ref=B_x,colab1=A_x,colab2=C_x,colab_action=[self.AB,self.AC,self.BC])
          TP3,FP3,FN3= self.get_TPFPFN(view =3,image_id=video_frame,det_content=det_content3,theta1 = RL_THETA1 ,theta2 = RL_THETA2,theta3 = RL_THETA3,ref=C_x,colab1=A_x,colab2=B_x,colab_action=[self.AB,self.AC,self.BC]) 

          TP1_all,FP1_all,FN1_all= self.get_TPFPFN(view =1,image_id=video_frame,det_content=det_content1,theta1 = RL_THETA1 ,theta2 = RL_THETA2,theta3 = RL_THETA3,ref=A_x,colab1=B_x,colab2=C_x,colab_action=[1,1,1])
          TP2_all,FP2_all,FN2_all= self.get_TPFPFN(view =2,image_id=video_frame,det_content=det_content2,theta1 = RL_THETA1 ,theta2 = RL_THETA2,theta3 = RL_THETA3,ref=B_x,colab1=A_x,colab2=C_x,colab_action=[1,1,1])
          TP3_all,FP3_all,FN3_all= self.get_TPFPFN(view =3,image_id=video_frame,det_content=det_content3,theta1 = RL_THETA1 ,theta2 = RL_THETA2,theta3 = RL_THETA3,ref=C_x,colab1=A_x,colab2=B_x,colab_action=[1,1,1]) 
          def calculate_metrics(tp, fp, fn):
            if tp + fp != 0:
                precision = (tp / (tp + fp)) * 100.0
            else:
                precision = 0.0
            if tp + fn != 0:
                recall = (2 * precision * (tp / (tp + fn))) / (precision + (tp / (tp + fn)))
                f1_score = (tp / (tp + fn)) * 100.0
            else:
                recall = 0.0
                f1_score = 0.0
            return precision, recall, f1_score
          
          if (300>video_frame>101):
            set1_temp_c1_tp += TP1
            set1_temp_c1_fp += FP1
            set1_temp_c1_fn += FN2
            set1_temp_c2_tp += TP2
            set1_temp_c2_fp += FP2
            set1_temp_c2_fn += FN2 
            set1_temp_c3_tp += TP3
            set1_temp_c3_fp += FP3
            set1_temp_c3_fn += FN3
            #print(set1_temp_c1_tp,set1_temp_c1_fp,set1_temp_c1_fn,set1_temp_c2_tp,set1_temp_c2_fp,set1_temp_c2_fn,set1_temp_c3_tp,set1_temp_c3_fp,set1_temp_c3_fn)
            #f11 = float(set1_temp_c1_tp)/(set1_temp_c1_tp+set1_temp_c1_fn)*100.0
            #p11= float(set1_temp_c1_tp)/(set1_temp_c1_tp+set1_temp_c1_fp)*100.0
            #r11= 2*p11*f11/(p11+f11)
            #p12= float(set1_temp_c2_tp)/(set1_temp_c2_tp+set1_temp_c2_fp)*100.0
            #f12 = float(set1_temp_c2_tp)/(set1_temp_c2_tp+set1_temp_c2_fn)*100.0
            #r12= 2*p12*f12/(p12+f12)
            #p13= float(set1_temp_c3_tp)/(set1_temp_c3_tp+set1_temp_c3_fp)*100.0
            #f13 = float(set1_temp_c3_tp)/(set1_temp_c3_tp+set1_temp_c3_fn)*100.0
            #r13= 2*p13*f13/(p13+f13)


          if (500>video_frame>301):
            set2_temp_c1_tp += TP1
            set2_temp_c1_fp += FP1
            set2_temp_c1_fn += FN2
            set2_temp_c2_tp += TP2
            set2_temp_c2_fp += FP2
            set2_temp_c2_fn += FN2 
            set2_temp_c3_tp += TP3
            set2_temp_c3_fp += FP3
            set2_temp_c3_fn += FN3
            if(set2_temp_c1_tp+set2_temp_c1_fp != 0): 
              p21= float(set2_temp_c1_tp)/(set2_temp_c1_tp+set2_temp_c1_fp)*100.0
            else:
              p21=0
            if(set2_temp_c1_tp+set2_temp_c1_fn != 0):
              f21 = float(set2_temp_c1_tp)/(set2_temp_c1_tp+set2_temp_c1_fn)*100.0
            else:
              f21=0
            if(p21+f21 != 0):
              r21= 2*p21*f21/(p21+f21)
            else:
              r21=0
            if(set2_temp_c2_tp+set2_temp_c2_fp != 0):
              p22= float(set2_temp_c2_tp)/(set2_temp_c2_tp+set2_temp_c2_fp)*100.0
            else: 
              p22=0 
            if(set2_temp_c2_tp+set2_temp_c2_fn != 0):
              f22 = float(set2_temp_c2_tp)/(set2_temp_c2_tp+set2_temp_c2_fn)*100.0
            else:
              f22=0
            if(p22+f22 != 0):
              r22= 2*p22*f22/(p22+f22)
            else:
              r22=0
            if(set2_temp_c3_tp+set2_temp_c3_fp != 0):
              p23= float(set2_temp_c3_tp)/(set2_temp_c3_tp+set2_temp_c3_fp)*100.0
            else:
              p23=0
            if(set2_temp_c3_tp+set2_temp_c3_fn != 0):
              f23 = float(set2_temp_c3_tp)/(set2_temp_c3_tp+set2_temp_c3_fn)*100.0
            else:
              f23=0
            if(p23+f23 != 0):
              r23= 2*p23*f23/(p23+f23)
            else:
              r23=0
          if(video_frame%1==0): 
            cv2.imwrite("frame_cam1.jpg",xx1)
            cv2.imwrite("frame_cam2.jpg",xx2)
            cv2.imwrite("frame_cam3.jpg",xx3)
            xxx1 = cv2.resize(xx1,(500,500))
            xxx2 = cv2.resize(xx2,(500,500))
            xxx3 = cv2.resize(xx3,(500,500))
            yy1 = cv2.resize(yy1,(500,500))
            yy2 = cv2.resize(yy2,(500,500))
            yy3 = cv2.resize(yy3,(500,500))

            canvas = np.zeros((1000, 1500), dtype=np.uint8)
            # Place xx1, xx2, xx3 in the first row
            canvas[:500, :500] = xxx1
            canvas[:500, 500:1000] = xxx2
            canvas[:500, 1000:] = xxx3
            # Place yy1, yy2, yy3 in the second row
            canvas[500:, :500] = yy1
            canvas[500:, 500:1000] = yy2
            canvas[500:, 1000:] = yy3

            #cv2.imwrite("./video/" ""+str(video_frame)+".jpg",canvas)
            folder_path = "./video/"
            subfolder_path = os.path.join(folder_path,str(iteration_number))
            # Create subfolder
            os.makedirs(subfolder_path, exist_ok=True)
            new_path = os.path.join(subfolder_path, str(video_frame))
            cv2.imwrite(new_path+".jpg",canvas)

            #stacked_images = np.stack((xx1, xx2, xx3), axis=-1)
            averaged_image = self.preprocess_image(xx1, xx2, xx3)
            
            state=averaged_image
            #print(averaged_image.shape)

            pp = np.transpose(averaged_image, (1, 2, 0))
            print(pp.shape)
            cv2.imwrite("frame_cam4.jpg",pp)
            #action_index=np.argmax([self.ai.Q[(tuple(state), ACTION_THETA1_INC)] ,self.ai.Q[(tuple(state), ACTION_THETA1_DEC)],self.ai.Q[(tuple(state), ACTION_THETA2_INC)] ,self.ai.Q[(tuple(state), ACTION_THETA2_DEC)],self.ai.Q[(tuple(state), ACTION_THETA3_INC)] ,self.ai.Q[(tuple(state), ACTION_THETA3_DEC)],self.ai.Q[(tuple(state), DO_NOTHING)]])
            #action = action_list[action_index]
            #print(dic_action)
          if(video_frame%10==0 ): # and video_frame<250
            action_index, _states = model.predict(state, deterministic=True)
            action_index= int(action_index)
            action_list=dic_action[action_index]
            str_exe_action = dic_action_index[action_index]
            dic_test_summery_action[action_index]+=1
            RL_THETA1 = RL_THETA1 +10*action_list[0]
            RL_THETA2 = RL_THETA2 +10*action_list[1]
            RL_THETA3 = RL_THETA3 +10*action_list[2]
            self.AB = action_list[3]
            self.AC = action_list[4]
            self.BC = action_list[5]
          #print(video_frame,"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",str_exe_action)
          dic_cam1_angle_variation[video_frame] = RL_THETA1
          dic_cam2_angle_variation[video_frame] = RL_THETA2
          dic_cam3_angle_variation[video_frame] = RL_THETA3
          print(self.AB+self.AC,self.AB+self.BC,self.AC+self.BC)
          dic_camA_colab_behavior[video_frame] = self.AB+self.AC
          dic_camB_colab_behavior[video_frame] = self.AB+self.BC
          dic_camC_colab_behavior[video_frame] = self.AC+self.BC
          #if(ENABLE_PRINT):
          print("executed best action is ",str_exe_action)  
          RL_cul_TP1+= TP1
          RL_cul_FP1+= FP1
          RL_cul_FN1+= FN1
          RL_cul_TP2+= TP2
          RL_cul_FP2+= FP2
          RL_cul_FN2+= FN2
          RL_cul_TP3+= TP3
          RL_cul_FP3+= FP3
          RL_cul_FN3+= FN3
          RL_cul_TP1_all += TP1_all
          RL_cul_FP1_all += FP1_all
          RL_cul_FN1_all += FN1_all
          RL_cul_TP2_all += TP2_all
          RL_cul_FP2_all += FP2_all
          RL_cul_FN2_all += FN2_all
          RL_cul_TP3_all += TP3_all
          RL_cul_FP3_all += FP3_all
          RL_cul_FN3_all += FN3_all


          if(TP1+FP1 != 0):
            current_c1_dqn_pricision = float(TP1)/(TP1+FP1)*100.0
          else:
            current_c1_dqn_pricision = 0

          if(TP2+FP2 != 0):
            current_c2_dqn_pricision = float(TP2)/(TP2+FP2)*100.0
          else:
            current_c2_dqn_pricision = 0

          if(TP3+FP3 != 0):
            current_c3_dqn_pricision = float(TP3)/(TP3+FP3)*100.0
          else:
            current_c3_dqn_pricision = 0

          if(TP1+FN1 != 0):
            current_c1_dqn_recall = float(TP1)/(TP1+FN1)*100.0
          else:
            current_c1_dqn_recall = 0
          if(TP2+FN2 != 0):
            current_c2_dqn_recall = float(TP2)/(TP2+FN2)*100.0
          else:
            current_c2_dqn_recall = 0
          if(TP3+FN3 != 0):
            current_c3_dqn_recall = float(TP3)/(TP3+FN3)*100.0
          else:
            current_c3_dqn_recall = 0
          if(current_c1_dqn_pricision+current_c1_dqn_recall):
            current_c1_dqn_F1Score = 2*current_c1_dqn_pricision*current_c1_dqn_recall/(current_c1_dqn_pricision+current_c1_dqn_recall)
          else:
            current_c1_dqn_F1Score = 0

          if(current_c2_dqn_pricision+current_c2_dqn_recall):
            current_c2_dqn_F1Score = 2*current_c2_dqn_pricision*current_c2_dqn_recall/(current_c2_dqn_pricision+current_c2_dqn_recall)
          else:
            current_c2_dqn_F1Score = 0

          if(current_c3_dqn_pricision+current_c3_dqn_recall):
            current_c3_dqn_F1Score = 2*current_c3_dqn_pricision*current_c3_dqn_recall/(current_c3_dqn_pricision+current_c3_dqn_recall)
          else:
            current_c3_dqn_F1Score = 0
##############################################################################
          if(TP1_all+FP1_all != 0):
            current_c1_colab_pricision = float(TP1_all)/(TP1_all+FP1_all)*100.0
          else:
            current_c1_colab_pricision = 0

          if(TP2_all+FP2_all != 0):
            current_c2_colab_pricision = float(TP2_all)/(TP2_all+FP2_all)*100.0
          else:
            current_c2_colab_pricision = 0

          if(TP3_all+FP3_all != 0):
            current_c3_colab_pricision = float(TP3_all)/(TP3_all+FP3_all)*100.0
          else:
            current_c3_colab_pricision = 0

          if(TP1_all+FN1_all != 0):
            current_c1_colab_recall = float(TP1_all)/(TP1_all+FN1_all)*100.0
          else:
            current_c1_colab_recall = 0
          if(TP2_all+FN2_all != 0):
            current_c2_colab_recall = float(TP2_all)/(TP2_all+FN2_all)*100.0
          else:
            current_c2_colab_recall = 0
          if(TP3_all+FN3_all != 0):
            current_c3_colab_recall = float(TP3_all)/(TP3_all+FN3_all)*100.0
          else:
            current_c3_colab_recall = 0
          if(current_c1_colab_pricision+current_c1_colab_recall):
            current_c1_colab_F1Score = 2*current_c1_colab_pricision*current_c1_colab_recall/(current_c1_colab_pricision+current_c1_colab_recall)
          else:
            current_c1_colab_F1Score = 0

          if(current_c2_colab_pricision+current_c2_colab_recall):
            current_c2_colab_F1Score = 2*current_c2_colab_pricision*current_c2_colab_recall/(current_c2_colab_pricision+current_c2_colab_recall)
          else:
            current_c2_colab_F1Score = 0

          if(current_c3_colab_pricision+current_c3_colab_recall):
            current_c3_colab_F1Score = 2*current_c3_colab_pricision*current_c3_colab_recall/(current_c3_colab_pricision+current_c3_colab_recall)
          else:
            current_c3_colab_F1Score = 0
##############################################################################
          dic_cam1_baseline_accuracy[video_frame] = [current_c1_baseline_F1Score]
          dic_cam2_baseline_accuracy[video_frame] = [current_c2_baseline_F1Score]
          dic_cam3_baseline_accuracy[video_frame] = [current_c3_baseline_F1Score]
          dic_cam1_DQN_accuracy[video_frame] = [current_c1_dqn_F1Score]
          dic_cam2_DQN_accuracy[video_frame] = [current_c2_dqn_F1Score]
          dic_cam3_DQN_accuracy[video_frame] = [current_c3_dqn_F1Score]

          dic_cam1_colab_accuracy[video_frame] = [current_c1_colab_F1Score]
          dic_cam2_colab_accuracy[video_frame] = [current_c2_colab_F1Score]
          dic_cam3_colab_accuracy[video_frame] = [current_c3_colab_F1Score]

          occlusion1_DQN_cam1 = self.get_occlusion(det_content1)
          occlusion2_DQN_cam2 = self.get_occlusion(det_content2)
          occlusion3_DQN_cam3 = self.get_occlusion(det_content3)
          
          dic_cam1_DQN_occlusion[video_frame] = occlusion1_DQN_cam1
          dic_cam2_DQN_occlusion[video_frame] = occlusion2_DQN_cam2
          dic_cam3_DQN_occlusion[video_frame] = occlusion3_DQN_cam3
      
        if(cul_TP1+cul_FP1 != 0):
          c1_baseline_pricision = float(cul_TP1)/(cul_TP1+cul_FP1)*100.0 
        else:
          c1_baseline_pricision = 0
        if(cul_TP2+cul_FP2 != 0):  
          c1_baseline_Recall = float(cul_TP1)/(cul_TP1+cul_FN1)*100.0
        else:
          c1_baseline_Recall = 0
        if(c1_baseline_pricision+c1_baseline_Recall != 0): 
          c1_baseline_F1Score = 2*c1_baseline_pricision*c1_baseline_Recall/(c1_baseline_pricision+c1_baseline_Recall)
        else:
          c1_baseline_F1Score = 0
        if(cul_TP2+cul_FP2 != 0):
          c2_baseline_pricision = float(cul_TP2)/(cul_TP2+cul_FP2)*100.0
        else:
          c2_baseline_pricision = 0
        if(cul_TP2+cul_FN2 != 0):
          c2_baseline_Recall = float(cul_TP2)/(cul_TP2+cul_FN2)*100.0
        else:
          c2_baseline_Recall=0
     
        if(c2_baseline_pricision+c2_baseline_Recall != 0):
           c2_baseline_F1Score = 2*c2_baseline_pricision*c2_baseline_Recall/(c2_baseline_pricision+c2_baseline_Recall)
        else:
           c2_baseline_F1Score = 0

        if(cul_TP3+cul_FP3 != 0):
           c3_baseline_pricision = float(cul_TP3)/(cul_TP3+cul_FP3)*100.0   
        else:
            c3_baseline_pricision = 0
        if(cul_TP3+cul_FN3 != 0):
           c3_baseline_Recall = float(cul_TP3)/(cul_TP3+cul_FN3)*100.0
        else:
          c3_baseline_Recall=0
        if(c3_baseline_pricision+c3_baseline_Recall != 0):
          c3_baseline_F1Score = 2*c3_baseline_pricision*c3_baseline_Recall/(c3_baseline_pricision+c3_baseline_Recall)
        else:
          c3_baseline_F1Score = 0

        Avg_baseline = (c1_baseline_F1Score+c2_baseline_F1Score+c3_baseline_F1Score)/3
        print("individual fscores are baesline {} {} {}".format(c1_baseline_F1Score,c2_baseline_F1Score,c3_baseline_F1Score))
        
        if(RL_cul_TP1+RL_cul_FP1 != 0): 
          c1_RL_pricision = float(RL_cul_TP1)/(RL_cul_TP1+RL_cul_FP1)*100.0  
        else:
          c1_RL_pricision = 0
        if(RL_cul_TP1+RL_cul_FN1 != 0):
          c1_RL_Recall = float(RL_cul_TP1)/(RL_cul_TP1+RL_cul_FN1)*100.0
        else:
          c1_RL_Recall = 0
        if(c1_RL_pricision+c1_RL_Recall != 0):
          c1_RL_F1Score = 2*c1_RL_pricision*c1_RL_Recall/(c1_RL_pricision+c1_RL_Recall)
        else:
          c1_RL_F1Score = 0

        if(RL_cul_TP2+RL_cul_FP2 != 0):
          c2_RL_pricision = float(RL_cul_TP2)/(RL_cul_TP2+RL_cul_FP2)*100.0   
        else:
          c2_RL_pricision = 0
        if(RL_cul_TP2+RL_cul_FN2 != 0):
          c2_RL_Recall = float(RL_cul_TP2)/(RL_cul_TP2+RL_cul_FN2)*100.0
        else:
          c2_RL_Recall = 0
        if(c2_RL_pricision+c2_RL_Recall != 0):
          c2_RL_F1Score = 2*c2_RL_pricision*c2_RL_Recall/(c2_RL_pricision+c2_RL_Recall)
        else:
          c2_RL_F1Score = 0

        if(RL_cul_TP3+RL_cul_FP3 != 0):
          c3_RL_pricision = float(RL_cul_TP3)/(RL_cul_TP3+RL_cul_FP3)*100.0   
        else:
          c3_RL_pricision = 0
        if(RL_cul_TP3+RL_cul_FN3 != 0):
          c3_RL_Recall = float(RL_cul_TP3)/(RL_cul_TP3+RL_cul_FN3)*100.0
        else:
          c3_RL_Recall = 0
        if(c3_RL_pricision+c3_RL_Recall != 0):
           c3_RL_F1Score = 2*c3_RL_pricision*c3_RL_Recall/(c3_RL_pricision+c3_RL_Recall)
        else:
            c3_RL_F1Score = 0
# this is for alll collaboration accuracy calculation
        if(RL_cul_TP1_all+RL_cul_FP1_all != 0): 
          c1_RL_pricision_all = float(RL_cul_TP1_all)/(RL_cul_TP1_all+RL_cul_FP1_all)*100.0  
        else:
          c1_RL_pricision_all = 0
        if(RL_cul_TP1_all+RL_cul_FN1_all != 0):
          c1_RL_Recall_all = float(RL_cul_TP1_all)/(RL_cul_TP1_all+RL_cul_FN1_all)*100.0
        else:
          c1_RL_Recall_all = 0
        if(c1_RL_pricision_all+c1_RL_Recall_all != 0):
          c1_RL_F1Score_all = 2*c1_RL_pricision_all*c1_RL_Recall_all/(c1_RL_pricision_all+c1_RL_Recall_all)
        else:
          c1_RL_F1Score_all = 0
        if(RL_cul_TP2_all+RL_cul_FP2_all != 0):
          c2_RL_pricision_all = float(RL_cul_TP2_all)/(RL_cul_TP2_all+RL_cul_FP2_all)*100.0   
        else:
          c2_RL_pricision_all = 0
        if(RL_cul_TP2_all+RL_cul_FN2_all != 0):
          c2_RL_Recall_all = float(RL_cul_TP2_all)/(RL_cul_TP2_all+RL_cul_FN2_all)*100.0
        else:
          c2_RL_Recall_all = 0
        if(c2_RL_pricision_all+c2_RL_Recall_all != 0):
          c2_RL_F1Score_all = 2*c2_RL_pricision_all*c2_RL_Recall_all/(c2_RL_pricision_all+c2_RL_Recall_all)
        else:
          c2_RL_F1Score_all = 0
        if(RL_cul_TP3_all+RL_cul_FP3_all != 0):
          c3_RL_pricision_all = float(RL_cul_TP3_all)/(RL_cul_TP3_all+RL_cul_FP3_all)*100.0   
        else:
          c3_RL_pricision_all = 0
        if(RL_cul_TP3_all+RL_cul_FN3_all != 0):
          c3_RL_Recall_all = float(RL_cul_TP3_all)/(RL_cul_TP3_all+RL_cul_FN3_all)*100.0
        else:
          c3_RL_Recall_all = 0
        if(c3_RL_pricision_all+c3_RL_Recall_all!= 0):
           c3_RL_F1Score_all = 2*c3_RL_pricision_all*c3_RL_Recall_all/(c3_RL_pricision_all+c3_RL_Recall_all)
        else:
            c3_RL_F1Score_all = 0
        Avg_RL = (c1_RL_F1Score+c2_RL_F1Score+c3_RL_F1Score)/3
        Avg_RL_all = (c1_RL_F1Score_all+c2_RL_F1Score_all+c3_RL_F1Score_all)/3
        print("individual fscores are dqn {} {} {}".format(c1_RL_F1Score,c2_RL_F1Score,c3_RL_F1Score))
        print("individual fscores are dqn all colab {} {} {}".format(c1_RL_F1Score_all,c2_RL_F1Score_all,c3_RL_F1Score_all))
        print("average  fsocre for baseline is {} and average fscore for RL is {} and two collab fscore is {}".format(Avg_baseline,Avg_RL,Avg_RL_all)) 
        #print("individual fscores are RL {} {} {}".format(c1_RL_F1Score,c2_RL_F1Score,c3_RL_F1Score))

        self.plot_occlusion(dic_cam1_base_occlusion,dic_cam2_base_occlusion,dic_cam3_base_occlusion,iteration_number,type_im="./plots/base_occlusion_"+str(iteration_number))
        self.plot_occlusion(dic_cam1_DQN_occlusion,dic_cam2_DQN_occlusion,dic_cam3_DQN_occlusion,iteration_number,type_im="./plots/DQN_occlusion_"+str(iteration_number))
        print(dic_cam2_angle_variation)
        self.plot_able_variation(dic_cam1_angle_variation,dic_cam2_angle_variation,dic_cam3_angle_variation,iteration_number,type_im="./plots/angle_variation_"+str(iteration_number))
        self.plot_actions(dic_test_summery_action,iteration_number,"Test")
        self.plot_accuracy_per_frame(dic_cam1_baseline_accuracy,dic_cam2_baseline_accuracy,dic_cam3_baseline_accuracy,dic_cam1_DQN_accuracy,dic_cam2_DQN_accuracy,dic_cam3_DQN_accuracy,iteration_number)
        #self.plot_avg_accuracy_per_frame(dic_cam1_baseline_accuracy,dic_cam2_baseline_accuracy,dic_cam3_baseline_accuracy,dic_cam1_DQN_accuracy,dic_cam2_DQN_accuracy,dic_cam3_DQN_accuracy,iteration_number)
        self.plot_avg_accuracy_per_frame_with_colab(dic_cam1_baseline_accuracy,dic_cam2_baseline_accuracy,dic_cam3_baseline_accuracy,dic_cam1_DQN_accuracy,dic_cam2_DQN_accuracy,dic_cam3_DQN_accuracy,dic_camA_colab_behavior,dic_camB_colab_behavior,dic_camC_colab_behavior,iteration_number)
        self.plot_avg_accuracy_per_frame_with_colab_pure(dic_cam1_colab_accuracy,dic_cam2_colab_accuracy,dic_cam3_colab_accuracy,dic_cam1_DQN_accuracy,dic_cam2_DQN_accuracy,dic_cam3_DQN_accuracy,dic_camA_colab_behavior,dic_camB_colab_behavior,dic_camC_colab_behavior,iteration_number)
        self.plot_avg_accuracy_per_frame_with_colab_three(dic_cam1_baseline_accuracy,dic_cam2_baseline_accuracy,dic_cam3_baseline_accuracy,dic_cam1_colab_accuracy,dic_cam2_colab_accuracy,dic_cam3_colab_accuracy,dic_cam1_DQN_accuracy,dic_cam2_DQN_accuracy,dic_cam3_DQN_accuracy,iteration_number)

        total_cam1_occlusion = sum(dic_cam1_base_occlusion.values())
        total_cam2_occlusion = sum(dic_cam2_base_occlusion.values())
        total_cam3_occlusion = sum(dic_cam3_base_occlusion.values())
        average_occlusion_base = (total_cam1_occlusion+total_cam2_occlusion+total_cam3_occlusion)/3

        #total_cam1_twoFscore = sum(dic_cam1_colab_accuracy.values())
        #total_cam2_twoFscore = sum(dic_cam2_colab_accuracy.values())
        #total_cam3_twoFscore = sum(dic_cam3_colab_accuracy.values())
        #average_twoFscore = (total_cam1_twoFscore+total_cam2_twoFscore+total_cam3_twoFscore)/3
        #print("twofscore accuracy is",average_twoFscore)
        

        total_cam1_occlusion = sum(dic_cam1_DQN_occlusion.values())
        total_cam2_occlusion = sum(dic_cam2_DQN_occlusion.values())
        total_cam3_occlusion = sum(dic_cam3_DQN_occlusion.values())
        average_occlusion_DQN = (total_cam1_occlusion+total_cam2_occlusion+total_cam3_occlusion)/3
        
        logger.info("best model iteration number: %s", iteration_number)
        logger.info("average  fsocre for baseline is: %s", Avg_baseline)
        logger.info("########this is result for accuracy###############")
        logger.info("average fscore for RL %s", Avg_RL)
        logger.info("two collab average fscore for RL %s", Avg_RL_all)
        logger.info("dic_outofbound %s", dic_outofbound)
        logger.info("########this is result for occulution result########")
        logger.info("baseline coverage value is : %s ",average_occlusion_base)
        logger.info("RL coverage value is : %s",average_occlusion_DQN)
        
        logger.info("_________________________________________________________________________________________________________")
        #print("100-300 and 300-400",r13,r23)

 #   self.humman, self.computer = True, False
    def new_initialize_priors(self,ann_path,view):
            #720
        with open(ann_path) as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]
        content_mots  = []
        #append_cnt=0
        for line in lines:
          content = line.split(' ')
          #print(content)
          if (int(content[6])==1):
              continue
          else:
           human_id=content[1]
           xmin = int((int(content[2])/img_width)*300)
           ymin = int((int(content[3])/img_height)*300)
           xmax = int((int(content[4])/img_width)*300)
           ymax = int((int(content[5])/img_height)*300)
           content_mot = [content[0],xmin,ymin,xmax,ymax]

           content_mots.append(content_mot)
        return content_mots
    def new_initialize_priors_2(self,ann_path,view):
            #720
        with open(ann_path) as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]
        content_mots  = []
        #append_cnt=0
        for line in lines:
          content = line.split(' ')
          #print(content)
          if (int(content[6])==1):
              continue
          else:
           frame_id=content[0]
           human_id=content[1]
           xmin = int((int(content[2])/img_width)*300)
           ymin = int((int(content[3])/img_height)*300)
           xmax = int((int(content[4])/img_width)*300)
           ymax = int((int(content[5])/img_height)*300)
           content_mot = [content[0],xmin,ymin,xmax,ymax,human_id]

           content_mots.append(content_mot)
        return content_mots
    def load_files_and_initilize_for_occlution_canculartion(self):
      dic_A_gt={}
      dic_B_gt={}
      dic_C_gt={}
      image_root_path='./YOLOV3/'+dataset_load_name+'/'
      for dic_name in ["A0","A10","A20","A30","A40","A50","A60","A70","A80","A90","B0","B10","B20","B30","B40","B50","B60","B70","B80","B90","C0","C10","C20","C30","C40","C50","C60","C70","C80","C90"]:
        foldername  = dic_name
        if(foldername[0]=="A"):
          view1 =1
        elif(foldername[0]=="B"):
          view1 =2
        elif(foldername[0]=="C"):
          view1 =3
        final_image_path_one = image_root_path +'/' +foldername + "/View_00"+str(view1)+'.txt'
        cam1_gt = self.new_initialize_priors(final_image_path_one,view1)
        if(foldername[0]=="A"):
          dic_A_gt[dic_name] = cam1_gt
        elif(foldername[0]=="B"):
          dic_B_gt[dic_name] = cam1_gt
        elif(foldername[0]=="C"):
          dic_C_gt[dic_name] = cam1_gt
      return dic_A_gt,dic_B_gt,dic_C_gt 
    def load_files_and_initilize_for_occlution_canculartion_2(self):
      dic_A_gt={}
      dic_B_gt={}
      dic_C_gt={}
      image_root_path='./YOLOV3/'+dataset_load_name+'/'
      for dic_name in ["A0","A10","A20","A30","A40","A50","A60","A70","A80","A90","B0","B10","B20","B30","B40","B50","B60","B70","B80","B90","C0","C10","C20","C30","C40","C50","C60","C70","C80","C90"]:
        foldername  = dic_name
        if(foldername[0]=="A"):
          view1 =1
        elif(foldername[0]=="B"):
          view1 =2
        elif(foldername[0]=="C"):
          view1 =3
        final_image_path_one = image_root_path +'/' +foldername + "/View_00"+str(view1)+'.txt'
        cam1_gt = self.new_initialize_priors_2(final_image_path_one,view1)
        if(foldername[0]=="A"):
          dic_A_gt[dic_name] = cam1_gt
        elif(foldername[0]=="B"):
          dic_B_gt[dic_name] = cam1_gt
        elif(foldername[0]=="C"):
          dic_C_gt[dic_name] = cam1_gt
      return dic_A_gt,dic_B_gt,dic_C_gt 
    def cal_overlap(self,A,B):
      angle1=A[1:]
      angle2=B[1:]
      aa = main_dic[angle1]
      bb = main_dic[angle2]
      if(A[0]=="A"):
          index1=0
      elif(A[0]=="B"):
          index1=2
      elif(A[0]=="C"):
          index1=1
      if(B[0]=="A"):
          index2=0
      elif(B[0]=="B"):
          index2=2
      elif(B[0]=="C"):
          index2=1
      for i in aa:
          dic1= i[0]
      for i in bb:
          dic2= i[0]
      overlapped=0
      for real_coords in dic1.keys():
          ee1=dic1[real_coords]
          ee2=dic2[real_coords]
          current_loc_val1=ee1[index1]
          current_loc_val2=ee2[index2]
          if(current_loc_val1 and current_loc_val2):
              overlapped+=1
      pp=int((40-5)/0.5)
      total_locations= pp*pp
      overlapping_porsion=overlapped/total_locations
      return overlapping_porsion
    def startGame_test(self, model,iteration_number,init_angle,hh,T):
      dic_test_summery_action={0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0, 39: 0, 40: 0, 41: 0, 42: 0, 43: 0, 44: 0, 45: 0, 46: 0, 47: 0, 48: 0, 49: 0, 50: 0, 51: 0, 52: 0, 53: 0, 54: 0, 55: 0, 56: 0, 57: 0, 58: 0, 59: 0, 60: 0, 61: 0, 62: 0, 63: 0, 64: 0, 65: 0, 66: 0, 67: 0, 68: 0, 69: 0, 70: 0, 71: 0, 72: 0, 73: 0, 74: 0, 75: 0, 76: 0, 77: 0, 78: 0, 79: 0, 80: 0, 81: 0, 82: 0, 83: 0, 84: 0, 85: 0, 86: 0, 87: 0, 88: 0, 89: 0, 90: 0, 91: 0, 92: 0, 93: 0, 94: 0, 95: 0, 96: 0, 97: 0, 98: 0, 99: 0, 100: 0, 101: 0, 102: 0, 103: 0, 104: 0, 105: 0, 106: 0, 107: 0, 108: 0, 109: 0, 110: 0, 111: 0, 112: 0, 113: 0, 114: 0, 115: 0, 116: 0, 117: 0, 118: 0, 119: 0, 120: 0, 121: 0, 122: 0, 123: 0, 124: 0, 125: 0, 126: 0, 127: 0, 128: 0, 129: 0, 130: 0, 131: 0, 132: 0, 133: 0, 134: 0, 135: 0, 136: 0, 137: 0, 138: 0, 139: 0, 140: 0, 141: 0, 142: 0, 143: 0, 144: 0, 145: 0, 146: 0, 147: 0, 148: 0, 149: 0, 150: 0, 151: 0, 152: 0, 153: 0, 154: 0, 155: 0, 156: 0, 157: 0, 158: 0, 159: 0, 160: 0, 161: 0, 162: 0, 163: 0, 164: 0, 165: 0, 166: 0, 167: 0, 168: 0, 169: 0, 170: 0, 171: 0, 172: 0, 173: 0, 174: 0, 175: 0, 176: 0, 177: 0, 178: 0, 179: 0, 180: 0, 181: 0, 182: 0, 183: 0, 184: 0, 185: 0, 186: 0, 187: 0, 188: 0, 189: 0, 190: 0, 191: 0, 192: 0, 193: 0, 194: 0, 195: 0, 196: 0, 197: 0, 198: 0, 199: 0, 200: 0, 201: 0, 202: 0, 203: 0, 204: 0, 205: 0, 206: 0, 207: 0, 208: 0, 209: 0, 210: 0, 211: 0, 212: 0, 213: 0, 214: 0, 215: 0}
      if (1): #if AI     
        self.reset()
        done = False
        # this is for the baseline accuracy calculation

        init_angle_THETA1 = init_angle[0]
        init_angle_THETA2 = init_angle[1]
        init_angle_THETA3 = init_angle[2]
        AB=0
        BC=0
        AC=0


        THETA1_LOWER_BOUND_ANGLE = 0
        THETA2_LOWER_BOUND_ANGLE = 0
        THETA3_LOWER_BOUND_ANGLE = 0
        cul_TP1,cul_FP1,cul_FN1 =0,0,0
        cul_TP2,cul_FP2,cul_FN2 =0,0,0
        cul_TP3,cul_FP3,cul_FN3 =0,0,0
        RL_cul_TP1,RL_cul_FP1,RL_cul_FN1 =0,0,0
        RL_cul_TP2,RL_cul_FP2,RL_cul_FN2 =0,0,0
        RL_cul_TP3,RL_cul_FP3,RL_cul_FN3 =0,0,0
        RL_cul_TP1_all,RL_cul_FP1_all,RL_cul_FN1_all =0,0,0
        RL_cul_TP2_all,RL_cul_FP2_all,RL_cul_FN2_all =0,0,0
        RL_cul_TP3_all,RL_cul_FP3_all,RL_cul_FN3_all =0,0,0

        RL_THETA1= init_angle_THETA1
        RL_THETA2= init_angle_THETA2
        RL_THETA3= init_angle_THETA3

        dic_cam1_base_occlusion={}
        dic_cam2_base_occlusion={}
        dic_cam3_base_occlusion={}
        dic_cam1_DQN_occlusion={}
        dic_cam2_DQN_occlusion={}
        dic_cam3_DQN_occlusion={}
        dic_cams_angle_variation={}
        dic_cam1_angle_variation={}
        dic_cam2_angle_variation={}
        dic_cam3_angle_variation={}
        dic_cam1_baseline_accuracy={}
        dic_cam2_baseline_accuracy={}
        dic_cam3_baseline_accuracy={}
        dic_cam1_DQN_accuracy={}
        dic_cam2_DQN_accuracy={}
        dic_cam3_DQN_accuracy={}

        dic_AB_overlap_variation=[]
        dic_AC_overlap_variation=[]
        dic_BC_overlap_variation=[]

        dic_cam_baseline_coverage = {}
        dic_cam_dqn_coverage = {}

        dic_outofbound={}
        dic_A_gt,dic_B_gt,dic_C_gt =self.load_files_and_initilize_for_occlution_canculartion()
        dic_A_gt_new,dic_B_gt_new,dic_C_gt_new = self.load_files_and_initilize_for_occlution_canculartion_2()
        cummulative_AB=0
        cummulative_BC=0
        cummulative_AC=0
        cummulative_detected_perframe_dqn=0
        cummulative_detected_perframe_base=0

        accumulated_v1_f1score=[]
        accumulated_v2_f1score=[]
        accumulated_v3_f1score=[]
        flame_list=[]
        for video_frame in range (0,500):
          base_line_THETA1= init_angle_THETA1
          base_line_THETA2= init_angle_THETA2
          base_line_THETA3= init_angle_THETA3
          A_x = "A"+str(base_line_THETA1)
          B_x = "B"+str(base_line_THETA2)
          C_x = "C"+str(base_line_THETA3)
          #print("current steering angle for baseline is ",A_x,B_x,C_x)
          base_covering_area = self.get_coverage(A_x,B_x,C_x)
          #start_time =time.time()
          yy1,det_content1 = self.get_people_base(view1= 1,view2= 2,view3= 3,value=video_frame,ref=A_x,colab1=B_x,colab2=C_x,colab_action=[0,0,0])
          yy2,det_content2 = self.get_people_base(view1= 2,view2= 1,view3= 3,value=video_frame,ref=B_x,colab1=A_x,colab2=C_x,colab_action=[0,0,0])
          yy3,det_content3 = self.get_people_base(view1= 3,view2= 1,view3= 2,value=video_frame,ref=C_x,colab1=A_x,colab2=B_x,colab_action=[0,0,0])
          #fps = (time.time() - start_time)*1000
          #print(f"time take for comple the get_people_base is : {fps:.2f} ms") 
          #yy1 = cv2.resize(yy1,(500,500))
          #yy2 = cv2.resize(yy2,(500,500))
          #yy3 = cv2.resize(yy3,(500,500))
          #start_time =time.time()
          TP1,FP1,FN1= self.get_TPFPFN(view =1,image_id=video_frame,det_content=det_content1,theta1 = base_line_THETA1 ,theta2 = base_line_THETA2,theta3 = base_line_THETA3,ref=A_x,colab1=B_x,colab2=C_x,colab_action=[0,0,0])
          TP2,FP2,FN2= self.get_TPFPFN(view =2,image_id=video_frame,det_content=det_content2,theta1 = base_line_THETA1 ,theta2 = base_line_THETA2,theta3 = base_line_THETA3,ref=B_x,colab1=A_x,colab2=C_x,colab_action=[0,0,0])
          TP3,FP3,FN3= self.get_TPFPFN(view =3,image_id=video_frame,det_content=det_content3,theta1 = base_line_THETA1 ,theta2 = base_line_THETA2,theta3 = base_line_THETA3,ref=C_x,colab1=A_x,colab2=B_x,colab_action=[0,0,0])
          
          
          #fps = (time.time() - start_time)*1000
          #print(f"time take for comple the get_TPFPFN is : {fps:.2f} ms") 
          cul_TP1+= TP1
          cul_FP1+= FP1
          cul_FN1+= FN1
          cul_TP2+= TP2
          cul_FP2+= FP2
          cul_FN2+= FN2
          cul_TP3+= TP3
          cul_FP3+= FP3
          cul_FN3+= FN3
          if(TP1+FP1 != 0):
            current_c1_baseline_pricision = float(TP1)/(TP1+FP1)*100.0
          else:
            current_c1_baseline_pricision = 0

          if(TP2+FP2 != 0):
            current_c2_baseline_pricision = float(TP2)/(TP2+FP2)*100.0
          else:
            current_c2_baseline_pricision = 0

          if(TP3+FP3 != 0):
            current_c3_baseline_pricision = float(TP3)/(TP3+FP3)*100.0
          else:
            current_c3_baseline_pricision = 0

          if(TP1+FN1 != 0):
            current_c1_baseline_recall = float(TP1)/(TP1+FN1)*100.0
          else:
            current_c1_baseline_recall = 0

          if(TP2+FN2 != 0):
            current_c2_baseline_recall = float(TP2)/(TP2+FN2)*100.0
          else:
            current_c2_baseline_recall = 0

          if(TP3+FN3 != 0):
            current_c3_baseline_recall = float(TP3)/(TP3+FN3)*100.0
          else:
            current_c3_baseline_recall = 0


          if(current_c1_baseline_pricision+current_c1_baseline_recall):
            current_c1_baseline_F1Score = 2*current_c1_baseline_pricision*current_c1_baseline_recall/(current_c1_baseline_pricision+current_c1_baseline_recall)
          else:
            current_c1_baseline_F1Score = 0

          if(current_c2_baseline_pricision+current_c2_baseline_recall):
            current_c2_baseline_F1Score = 2*current_c2_baseline_pricision*current_c2_baseline_recall/(current_c2_baseline_pricision+current_c2_baseline_recall)
          else:
            current_c2_baseline_F1Score = 0

          if(current_c3_baseline_pricision+current_c3_baseline_recall):
            current_c3_baseline_F1Score = 2*current_c3_baseline_pricision*current_c3_baseline_recall/(current_c3_baseline_pricision+current_c3_baseline_recall)
          else:
            current_c3_baseline_F1Score = 0
          image_root_path='./YOLOV3/'+dataset_load_name+'/'
          foldername  = A_x
          view1 =1
          final_image_path_one = image_root_path +'/' +foldername + "/View_00"+str(view1)+'.txt'
          cam1_gt = dic_A_gt[foldername]
          # below code is to temmporary cal culate the global detections
          cam1_gt_new = dic_A_gt_new[foldername]
          all_people_id=[int(x) for x in range(11)]
          def iou(bb_det,bb_gt):
            #print('inside iou',bb_det,bb_gt)
            xx1 = np.maximum(bb_det[0],bb_gt[0])
            yy1 = np.maximum(bb_det[1],bb_gt[1])
            xx2 = np.minimum(bb_det[2],bb_gt[2])
            yy2 = np.minimum(bb_det[3],bb_gt[3])
            w = np.maximum(0.,xx2 - xx1)
            h = np.maximum(0.,yy2 - yy1)
            wh = w * h
            o = wh / ((bb_det[2]-bb_det[0])*(bb_det[3]-bb_det[1])
                    + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
            return o 
          
          #cam1_gt = self.new_initialize_priors(final_image_path_one,view1)
          val_content1 = [x[1:] for x in cam1_gt if int(x[0])==video_frame]
          gt_content_1=[x[1:-1] for x in cam1_gt_new if int(x[0])==video_frame]
          gt_content_id_1 = [x[-1] for x in cam1_gt_new if int(x[0])==video_frame]

          foldername  = B_x
          view2 =2
          final_image_path_one = image_root_path +'/' +foldername + "/View_00"+str(view2)+'.txt'
          #cam2_gt = self.new_initialize_priors(final_image_path_one,view2)
          cam2_gt =dic_B_gt[foldername] 
          cam2_gt_new = dic_B_gt_new[foldername]
          val_content2 = [x[1:] for x in cam2_gt if int(x[0])==video_frame]
          gt_content_2=[x[1:-1] for x in cam2_gt_new if int(x[0])==video_frame]
          gt_content_id_2 = [x[-1] for x in cam2_gt_new if int(x[0])==video_frame]
          
          foldername  = C_x
          view3 =3
          final_image_path_one = image_root_path +'/' +foldername + "/View_00"+str(view3)+'.txt'
          #cam3_gt = self.new_initialize_priors(final_image_path_one,view3)
          cam3_gt  = dic_C_gt[foldername] 
          cam3_gt_new = dic_C_gt_new[foldername]
          val_content3 = [x[1:] for x in cam3_gt if int(x[0])==video_frame]
          gt_content_3=[x[1:-1] for x in cam3_gt_new if int(x[0])==video_frame]
          gt_content_id_3 = [x[-1] for x in cam3_gt_new if int(x[0])==video_frame]

          total_number_perframe= len(all_people_id)
          iou_matrix =  np.zeros((len(det_content1),len(gt_content_1)),dtype=np.float32) 
          for d,det in enumerate(det_content1): 
            #bb_det = [int(det[2]),int(det[3]),int(det[2])+int(det[4]),int(det[3])+int(det[5])]
            for g,gt in enumerate(gt_content_1):
                #bb_gt = [int(gt[2]),int(gt[3]),int(gt[2])+int(gt[4]),int(gt[3])+int(gt[5])]
                iou_matrix[d,g] = iou(det,gt)
                #print(d,g,iou_matrix[d,g])
          #print('iou_matrix',iou_matrix)
          matched_indices = linear_assignment(-iou_matrix)
          for i in matched_indices:
            gt_id = int(i[1])
            if gt_id in all_people_id:
              all_people_id.remove(gt_id)
          iou_matrix =  np.zeros((len(det_content2),len(gt_content_2)),dtype=np.float32) 
          for d,det in enumerate(det_content2): 
            #bb_det = [int(det[2]),int(det[3]),int(det[2])+int(det[4]),int(det[3])+int(det[5])]
            for g,gt in enumerate(gt_content_2):
                #bb_gt = [int(gt[2]),int(gt[3]),int(gt[2])+int(gt[4]),int(gt[3])+int(gt[5])]
                iou_matrix[d,g] = iou(det,gt)
                #print(d,g,iou_matrix[d,g])
          #print('iou_matrix',iou_matrix)
          matched_indices = linear_assignment(-iou_matrix)
          for i in matched_indices:
            gt_id = int(i[1])
            if gt_id in all_people_id:
              all_people_id.remove(gt_id)
          iou_matrix =  np.zeros((len(det_content3),len(gt_content_3)),dtype=np.float32) 
          for d,det in enumerate(det_content3): 
            #bb_det = [int(det[2]),int(det[3]),int(det[2])+int(det[4]),int(det[3])+int(det[5])]
            for g,gt in enumerate(gt_content_3):
                #bb_gt = [int(gt[2]),int(gt[3]),int(gt[2])+int(gt[4]),int(gt[3])+int(gt[5])]
                iou_matrix[d,g] = iou(det,gt)
                #print(d,g,iou_matrix[d,g])
          #print('iou_matrix',iou_matrix)
          matched_indices = linear_assignment(-iou_matrix)
          for i in matched_indices:
            gt_id = int(i[1])
            if gt_id in all_people_id:
              all_people_id.remove(gt_id)   
          #print(all_people_id)
          total_undetected_perframe= len(all_people_id)
          total_detected_perframe_base= total_number_perframe-total_undetected_perframe
          cummulative_detected_perframe_base+=total_detected_perframe_base 
     
          
          occlusion1_base_cam1 = self.get_occlusion(val_content1)
          occlusion2_base_cam2 = self.get_occlusion(val_content2)
          occlusion3_base_cam3 = self.get_occlusion(val_content3)

          #occlusion1_base_cam1 = self.get_occlusion(det_content1)
          #occlusion2_base_cam2 = self.get_occlusion(det_content2)
          #occlusion3_base_cam3 = self.get_occlusion(det_content3)
        

          dic_cam1_base_occlusion[video_frame] = occlusion1_base_cam1
          dic_cam2_base_occlusion[video_frame] = occlusion2_base_cam2
          dic_cam3_base_occlusion[video_frame] = occlusion3_base_cam3

          if(RL_THETA1>90):
               RL_THETA1 = 90
          elif(RL_THETA1<0):
              RL_THETA1 = 0

          if(RL_THETA2>90):
               RL_THETA2 = 90
          elif(RL_THETA2<0):
              RL_THETA2 = 0

          if(RL_THETA3>90):
               RL_THETA3 = 90
          elif(RL_THETA3<0):
              RL_THETA3 = 0


          A_x = "A"+str(RL_THETA1)
          B_x = "B"+str(RL_THETA2)
          C_x = "C"+str(RL_THETA3)
          #print("current steering angle for RL is ",A_x,B_x,C_x)
          
          
          #start_time=time.time()
          xx1,det_content1 = self.get_people(view1= 1,view2= 2,view3= 3,value=video_frame,ref=A_x,colab1=B_x,colab2=C_x,colab_action=[AB,AC,BC]) #[AB,AC,BC]
          xx2,det_content2 = self.get_people(view1= 2,view2= 1,view3= 3,value=video_frame,ref=B_x,colab1=A_x,colab2=C_x,colab_action=[AB,AC,BC])
          xx3,det_content3 = self.get_people(view1= 3,view2= 1,view3= 2,value=video_frame,ref=C_x,colab1=A_x,colab2=B_x,colab_action=[AB,AC,BC])
          ppc = xx1.copy()
          ppc=cv2.resize(ppc,(300,300))
          #fps = (time.time() - start_time)*1000
          #print(f"time take for comple the get_people is : {fps:.2f} ms") 
          dqn_covering_area = self.get_coverage(A_x,B_x,C_x)
          #start_time=time.time()
          TP1,FP1,FN1= self.get_TPFPFN(view =1,image_id=video_frame,det_content=det_content1,theta1 = RL_THETA1 ,theta2 = RL_THETA2,theta3 = RL_THETA3,ref=A_x,colab1=B_x,colab2=C_x,colab_action=[AB,AC,BC])
          TP2,FP2,FN2= self.get_TPFPFN(view =2,image_id=video_frame,det_content=det_content2,theta1 = RL_THETA1 ,theta2 = RL_THETA2,theta3 = RL_THETA3,ref=B_x,colab1=A_x,colab2=C_x,colab_action=[AB,AC,BC])
          TP3,FP3,FN3= self.get_TPFPFN(view =3,image_id=video_frame,det_content=det_content3,theta1 = RL_THETA1 ,theta2 = RL_THETA2,theta3 = RL_THETA3,ref=C_x,colab1=A_x,colab2=B_x,colab_action=[AB,AC,BC])  

          TP1_all,FP1_all,FN1_all= self.get_TPFPFN(view =1,image_id=video_frame,det_content=det_content1,theta1 = RL_THETA1 ,theta2 = RL_THETA2,theta3 = RL_THETA3,ref=A_x,colab1=B_x,colab2=C_x,colab_action=[1,1,1])
          TP2_all,FP2_all,FN2_all= self.get_TPFPFN(view =2,image_id=video_frame,det_content=det_content2,theta1 = RL_THETA1 ,theta2 = RL_THETA2,theta3 = RL_THETA3,ref=B_x,colab1=A_x,colab2=C_x,colab_action=[1,1,1])
          TP3_all,FP3_all,FN3_all= self.get_TPFPFN(view =3,image_id=video_frame,det_content=det_content3,theta1 = RL_THETA1 ,theta2 = RL_THETA2,theta3 = RL_THETA3,ref=C_x,colab1=A_x,colab2=B_x,colab_action=[1,1,1]) 
          #print(f"time take for comple  next the get_TPFPFN is : {fps:.2f} ms") 
          if(video_frame%1==0): 

            #stacked_images = np.stack((xx1, xx2, xx3), axis=-1)
            averaged_image = self.preprocess_image(xx1, xx2, xx3)
            
            state=averaged_image
            #print(averaged_image.shape)
            pp = np.transpose(averaged_image, (1, 2, 0))
            
            #action = action_list[action_index]
            #print(dic_action)
          #if(AB==1):
          AB_overlap = self.cal_overlap("A"+str(RL_THETA1),"B"+str(RL_THETA2))
          #if(AC==1):
          AC_overlap = self.cal_overlap("A"+str(RL_THETA1),"C"+str(RL_THETA3))
          #print(AC_overlap,"A"+str(RL_THETA1),"C"+str(RL_THETA3))
          #if(BC==1):
          BC_overlap = self.cal_overlap("B"+str(RL_THETA2),"C"+str(RL_THETA3))
          if(video_frame%T==0):
            action_index, _states = model.predict(state, deterministic=True)
            action_index= int(action_index)
            action_list=dic_action[action_index]
            str_exe_action = dic_action_index[action_index]
            dic_test_summery_action[action_index]+=1
            #if(RL_THETA1>90):
            #   RL_THETA1 = 90
            #elif(RL_THETA1<0):
            #  RL_THETA1 = 0
            #else:
            RL_THETA1 = RL_THETA1 +10*action_list[0]

            #if(RL_THETA2>90):
            #   RL_THETA2 = 90
            #elif(RL_THETA2<0):
            #  RL_THETA2 = 0
            #else:
            RL_THETA2 = RL_THETA2 +10*action_list[1]

            #if(RL_THETA3>90):
            #   RL_THETA3 = 90
            #elif(RL_THETA3<0):
            #  RL_THETA3 = 0
            #else:
            RL_THETA3 = RL_THETA3 +10*action_list[2]

            #RL_THETA2 = RL_THETA2 +10*action_list[1]
            #RL_THETA3 = RL_THETA3 +10*action_list[2]
            AB = action_list[3]
            AC = action_list[4]
            BC = action_list[5]
            
            
            cummulative_AB+=AB
            cummulative_BC+=BC
            cummulative_AC+=AC
          dic_AB_overlap_variation.append([AB,AB_overlap])
          dic_AC_overlap_variation.append([AC,AC_overlap])
          dic_BC_overlap_variation.append([BC,BC_overlap])
          #print(RL_THETA1,RL_THETA2,RL_THETA3,cummulative_AB,cummulative_BC,cummulative_AC)
          
          dic_cams_angle_variation[video_frame] = [RL_THETA1,RL_THETA2,RL_THETA3]
          dic_cam1_angle_variation[video_frame] = RL_THETA1
          dic_cam2_angle_variation[video_frame] = RL_THETA2
          dic_cam3_angle_variation[video_frame] = RL_THETA3
          RL_cul_TP1+= TP1
          RL_cul_FP1+= FP1
          RL_cul_FN1+= FN1
          RL_cul_TP2+= TP2
          RL_cul_FP2+= FP2
          RL_cul_FN2+= FN2
          RL_cul_TP3+= TP3
          RL_cul_FP3+= FP3
          RL_cul_FN3+= FN3
          RL_cul_TP1_all += TP1_all
          RL_cul_FP1_all += FP1_all
          RL_cul_FN1_all += FN1_all
          RL_cul_TP2_all += TP2_all
          RL_cul_FP2_all += FP2_all
          RL_cul_FN2_all += FN2_all
          RL_cul_TP3_all += TP3_all
          RL_cul_FP3_all += FP3_all
          RL_cul_FN3_all += FN3_all
          if(TP1+FP1 != 0):
            current_c1_dqn_pricision = float(TP1)/(TP1+FP1)*100.0
          else:
            current_c1_dqn_pricision = 0

          if(TP2+FP2 != 0):
            current_c2_dqn_pricision = float(TP2)/(TP2+FP2)*100.0
          else:
            current_c2_dqn_pricision = 0

          if(TP3+FP3 != 0):
            current_c3_dqn_pricision = float(TP3)/(TP3+FP3)*100.0
          else:
            current_c3_dqn_pricision = 0

          if(TP1+FN1 != 0):
            current_c1_dqn_recall = float(TP1)/(TP1+FN1)*100.0
          else:
            current_c1_dqn_recall = 0
          if(TP2+FN2 != 0):
            current_c2_dqn_recall = float(TP2)/(TP2+FN2)*100.0
          else:
            current_c2_dqn_recall = 0
          if(TP3+FN3 != 0):
            current_c3_dqn_recall = float(TP3)/(TP3+FN3)*100.0
          else:
            current_c3_dqn_recall = 0
          if(current_c1_dqn_pricision+current_c1_dqn_recall):
            current_c1_dqn_F1Score = 2*current_c1_dqn_pricision*current_c1_dqn_recall/(current_c1_dqn_pricision+current_c1_dqn_recall)
          else:
            current_c1_dqn_F1Score = 0

          if(current_c2_dqn_pricision+current_c2_dqn_recall):
            current_c2_dqn_F1Score = 2*current_c2_dqn_pricision*current_c2_dqn_recall/(current_c2_dqn_pricision+current_c2_dqn_recall)
          else:
            current_c2_dqn_F1Score = 0

          if(current_c3_dqn_pricision+current_c3_dqn_recall):
            current_c3_dqn_F1Score = 2*current_c3_dqn_pricision*current_c3_dqn_recall/(current_c3_dqn_pricision+current_c3_dqn_recall)
          else:
            current_c3_dqn_F1Score = 0
          #print(video_frame,"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",str_exe_action,RL_THETA1,RL_THETA2,RL_THETA3)
          #print("baseline cam1 f1 {}, cam2 f1 {},cam 3 f1 {}".format(current_c1_baseline_F1Score,current_c2_baseline_F1Score,current_c3_baseline_F1Score))
          #print("DQn cam1 f1 {}, cam2 f1 {},cam 3 f1 {}".format(current_c1_dqn_F1Score,current_c2_dqn_F1Score,current_c3_dqn_F1Score))
          dic_cam1_baseline_accuracy[video_frame] = [current_c1_baseline_F1Score]
          dic_cam2_baseline_accuracy[video_frame] = [current_c2_baseline_F1Score]
          dic_cam3_baseline_accuracy[video_frame] = [current_c3_baseline_F1Score]
          dic_cam_baseline_coverage[video_frame] = base_covering_area
          dic_cam1_DQN_accuracy[video_frame] = [current_c1_dqn_F1Score]
          dic_cam2_DQN_accuracy[video_frame] = [current_c2_dqn_F1Score]
          dic_cam3_DQN_accuracy[video_frame] = [current_c3_dqn_F1Score]
          #print(video_frame,"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",str_exe_action,RL_THETA1,RL_THETA2,RL_THETA3)
          dic_cam_dqn_coverage[video_frame] = dqn_covering_area

          all_people_id=[int(x) for x in range(11)]
          total_number_perframe_dqn =len(all_people_id)
          image_root_path='./YOLOV3/'+dataset_load_name+'/'
          foldername  = A_x
          view1 =1
          final_image_path_one = image_root_path +'/' +foldername + "/View_00"+str(view1)+'.txt'
         # cam1_gt = self.new_initialize_priors(final_image_path_one,view1)
          cam1_gt  = dic_A_gt[foldername] 
          cam1_gt_new = dic_A_gt_new[foldername]
          val_content1 = [x[1:] for x in cam1_gt if int(x[0])==video_frame]
          gt_content_1=[x[1:-1] for x in cam1_gt_new if int(x[0])==video_frame]
          gt_content_id_1 = [x[-1] for x in cam1_gt_new if int(x[0])==video_frame]

          #print(val_content1)
          #print(det_content1)
          foldername  = B_x
          view2 =2
          final_image_path_one = image_root_path +'/' +foldername + "/View_00"+str(view2)+'.txt'
          #cam2_gt = self.new_initialize_priors(final_image_path_one,view2)
          cam2_gt  = dic_B_gt[foldername] 
          cam2_gt_new = dic_B_gt_new[foldername]
          val_content2 = [x[1:] for x in cam2_gt if int(x[0])==video_frame]
          gt_content_2=[x[1:-1] for x in cam2_gt_new if int(x[0])==video_frame]
          gt_content_id_2= [x[-1] for x in cam2_gt_new if int(x[0])==video_frame]

          foldername  = C_x
          view3 =3
          final_image_path_one = image_root_path +'/' +foldername + "/View_00"+str(view3)+'.txt'
          #cam3_gt = self.new_initialize_priors(final_image_path_one,view3)
          cam3_gt  = dic_C_gt[foldername] 
          cam3_gt_new = dic_C_gt_new[foldername]
          val_content3 = [x[1:] for x in cam3_gt if int(x[0])==video_frame]
          gt_content_3=[x[1:-1] for x in cam3_gt_new if int(x[0])==video_frame]
          gt_content_id_3 = [x[-1] for x in cam3_gt_new if int(x[0])==video_frame]

          #for bbox in gt_content_2:
          #   xmin=int(bbox[0])
          #   ymin=int(bbox[1])
          #   xmax=int(bbox[2])
          #   ymax=int(bbox[3])
          #   ppc=cv2.rectangle(ppc, (xmin, ymin), (xmax, ymax), (0,255,0), 1)
          #ppc=cv2.resize(ppc,(500,500))
          #cv2.imwrite("frame_cam1.jpg",ppc)
          iou_matrix =  np.zeros((len(det_content1),len(gt_content_1)),dtype=np.float32) 
          for d,det in enumerate(det_content1): 
            #bb_det = [int(det[2]),int(det[3]),int(det[2])+int(det[4]),int(det[3])+int(det[5])]
            for g,gt in enumerate(gt_content_1):
                #bb_gt = [int(gt[2]),int(gt[3]),int(gt[2])+int(gt[4]),int(gt[3])+int(gt[5])]
                iou_matrix[d,g] = iou(det,gt)
                #print(d,g,iou_matrix[d,g])
         # print('iou_matrix',iou_matrix)
          matched_indices = linear_assignment(-iou_matrix)
         # print(matched_indices)
          for i in matched_indices:
            gt_id = int(i[1])
            if gt_id in all_people_id:
              all_people_id.remove(gt_id)
          iou_matrix =  np.zeros((len(det_content2),len(gt_content_2)),dtype=np.float32) 
          for d,det in enumerate(det_content2): 
            #bb_det = [int(det[2]),int(det[3]),int(det[2])+int(det[4]),int(det[3])+int(det[5])]
            for g,gt in enumerate(gt_content_2):
                #bb_gt = [int(gt[2]),int(gt[3]),int(gt[2])+int(gt[4]),int(gt[3])+int(gt[5])]
                iou_matrix[d,g] = iou(det,gt)
                #print(d,g,iou_matrix[d,g])
         # print('iou_matrix',iou_matrix)
          matched_indices = linear_assignment(-iou_matrix)
          #print(matched_indices)
          for i in matched_indices:
            gt_id = int(i[1])
            if gt_id in all_people_id:
              all_people_id.remove(gt_id)
          iou_matrix =  np.zeros((len(det_content3),len(gt_content_3)),dtype=np.float32) 
          for d,det in enumerate(det_content3): 
            #bb_det = [int(det[2]),int(det[3]),int(det[2])+int(det[4]),int(det[3])+int(det[5])]
            for g,gt in enumerate(gt_content_3):
                #bb_gt = [int(gt[2]),int(gt[3]),int(gt[2])+int(gt[4]),int(gt[3])+int(gt[5])]
                iou_matrix[d,g] = iou(det,gt)
                #print(d,g,iou_matrix[d,g])
          #print('iou_matrix',iou_matrix)
          matched_indices = linear_assignment(-iou_matrix)
          #print(matched_indices)
          for i in matched_indices:
            gt_id = int(i[1])
            if gt_id in all_people_id:
              all_people_id.remove(gt_id) 
          total_undetected_perframe_dqn= len(all_people_id)
          total_detected_perframe_dqn= total_number_perframe_dqn-total_undetected_perframe_dqn 
          cummulative_detected_perframe_dqn += total_detected_perframe_dqn
          #break

          occlusion1_DQN_cam1 = self.get_occlusion(val_content1)
          occlusion2_DQN_cam2 = self.get_occlusion(val_content2)
          occlusion3_DQN_cam3 = self.get_occlusion(val_content3)  #det_content3
          dic_cam1_DQN_occlusion[video_frame] = occlusion1_DQN_cam1
          dic_cam2_DQN_occlusion[video_frame] = occlusion2_DQN_cam2
          dic_cam3_DQN_occlusion[video_frame] = occlusion3_DQN_cam3
          
        #print("cul_TP2 {} ,cul_FP2 {} ,cul_FN2 {}".format(cul_TP2,cul_FP2,cul_FN2)) 
        if(cul_TP1+cul_FP1 != 0):  
          c1_baseline_pricision = float(cul_TP1)/(cul_TP1+cul_FP1)*100.0   
        else:
          c1_baseline_pricision = 0
        if(cul_TP1+cul_FN1 != 0):
          c1_baseline_Recall = float(cul_TP1)/(cul_TP1+cul_FN1)*100.0
        else:
          c1_baseline_Recall = 0
        if(c1_baseline_pricision+c1_baseline_Recall != 0):
          c1_baseline_F1Score = 2*c1_baseline_pricision*c1_baseline_Recall/(c1_baseline_pricision+c1_baseline_Recall)
        else:
          c1_baseline_F1Score = 0
        if(cul_TP2+cul_FP2 != 0):
         c2_baseline_pricision = float(cul_TP2)/(cul_TP2+cul_FP2)*100.0 
        else:
          c2_baseline_pricision = 0
        if(cul_TP2+cul_FN2 != 0):  
          c2_baseline_Recall = float(cul_TP2)/(cul_TP2+cul_FN2)*100.0
        else:
          c2_baseline_Recall = 0
        if(c2_baseline_pricision+c2_baseline_Recall!= 0):
         c2_baseline_F1Score = 2*c2_baseline_pricision*c2_baseline_Recall/(c2_baseline_pricision+c2_baseline_Recall)
        else:
          c2_baseline_F1Score = 0
        if(cul_TP3+cul_FP3 != 0):
          c3_baseline_pricision = float(cul_TP3)/(cul_TP3+cul_FP3)*100.0   
        else:
          c3_baseline_pricision = 0
        if(cul_TP3+cul_FN3 != 0):
          c3_baseline_Recall = float(cul_TP3)/(cul_TP3+cul_FN3)*100.0
        else:
          c3_baseline_Recall = 0
        if(c3_baseline_pricision+c3_baseline_Recall != 0):
          c3_baseline_F1Score = 2*c3_baseline_pricision*c3_baseline_Recall/(c3_baseline_pricision+c3_baseline_Recall)
        else:
          c3_baseline_F1Score = 0
        Avg_baseline = (c1_baseline_F1Score+c2_baseline_F1Score+c3_baseline_F1Score)/3
        print('baseline accuracies are',c1_baseline_F1Score,c2_baseline_F1Score,c3_baseline_F1Score)
        print('rl accuracies are',RL_cul_TP1+RL_cul_FP1,RL_cul_TP2+RL_cul_FP2,RL_cul_TP3+RL_cul_FP3)
        if(RL_cul_TP1+RL_cul_FP1 != 0):
         c1_RL_pricision = float(RL_cul_TP1)/(RL_cul_TP1+RL_cul_FP1)*100.0  
        else:
          c1_RL_pricision = 0
        if(RL_cul_TP1+RL_cul_FN1 != 0):
         c1_RL_Recall = float(RL_cul_TP1)/(RL_cul_TP1+RL_cul_FN1)*100.0
        else:
          c1_RL_Recall = 0
        if(c1_RL_pricision+c1_RL_Recall !=0): 
          c1_RL_F1Score = 2*c1_RL_pricision*c1_RL_Recall/(c1_RL_pricision+c1_RL_Recall)
        else:
          c1_RL_F1Score = 0

        if(RL_cul_TP2+RL_cul_FP2 != 0):
          c2_RL_pricision = float(RL_cul_TP2)/(RL_cul_TP2+RL_cul_FP2)*100.0   
        else:
          c2_RL_pricision = 0
        if(RL_cul_TP2+RL_cul_FN2 != 0):
          c2_RL_Recall = float(RL_cul_TP2)/(RL_cul_TP2+RL_cul_FN2)*100.0
        else:
          c2_RL_Recall = 0

        if(c2_RL_pricision+c2_RL_Recall != 0):
         c2_RL_F1Score = 2*c2_RL_pricision*c2_RL_Recall/(c2_RL_pricision+c2_RL_Recall)
        else:
          c2_RL_F1Score = 0

        if(RL_cul_TP3+RL_cul_FP3 != 0):
          c3_RL_pricision = float(RL_cul_TP3)/(RL_cul_TP3+RL_cul_FP3)*100.0   
        else:
          c3_RL_pricision = 0
        if(RL_cul_TP3+RL_cul_FN3 != 0):
          c3_RL_Recall = float(RL_cul_TP3)/(RL_cul_TP3+RL_cul_FN3)*100.0
        else:
          c3_RL_Recall = 0
        if(c3_RL_pricision+c3_RL_Recall !=0):
          c3_RL_F1Score = 2*c3_RL_pricision*c3_RL_Recall/(c3_RL_pricision+c3_RL_Recall)
        else:
          c3_RL_F1Score = 0
# this is for alll collaboration accuracy calculation
        if(RL_cul_TP1_all+RL_cul_FP1_all != 0): 
          c1_RL_pricision_all = float(RL_cul_TP1_all)/(RL_cul_TP1_all+RL_cul_FP1_all)*100.0  
        else:
          c1_RL_pricision_all = 0
        if(RL_cul_TP1_all+RL_cul_FN1_all != 0):
          c1_RL_Recall_all = float(RL_cul_TP1_all)/(RL_cul_TP1_all+RL_cul_FN1_all)*100.0
        else:
          c1_RL_Recall_all = 0
        if(c1_RL_pricision_all+c1_RL_Recall_all != 0):
          c1_RL_F1Score_all = 2*c1_RL_pricision_all*c1_RL_Recall_all/(c1_RL_pricision_all+c1_RL_Recall_all)
        else:
          c1_RL_F1Score_all = 0
        if(RL_cul_TP2_all+RL_cul_FP2_all != 0):
          c2_RL_pricision_all = float(RL_cul_TP2_all)/(RL_cul_TP2_all+RL_cul_FP2_all)*100.0   
        else:
          c2_RL_pricision_all = 0
        if(RL_cul_TP2_all+RL_cul_FN2_all != 0):
          c2_RL_Recall_all = float(RL_cul_TP2_all)/(RL_cul_TP2_all+RL_cul_FN2_all)*100.0
        else:
          c2_RL_Recall_all = 0
        if(c2_RL_pricision_all+c2_RL_Recall_all != 0):
          c2_RL_F1Score_all = 2*c2_RL_pricision_all*c2_RL_Recall_all/(c2_RL_pricision_all+c2_RL_Recall_all)
        else:
          c2_RL_F1Score_all = 0
        if(RL_cul_TP3_all+RL_cul_FP3_all != 0):
          c3_RL_pricision_all = float(RL_cul_TP3_all)/(RL_cul_TP3_all+RL_cul_FP3_all)*100.0   
        else:
          c3_RL_pricision_all = 0
        if(RL_cul_TP3_all+RL_cul_FN3_all != 0):
          c3_RL_Recall_all = float(RL_cul_TP3_all)/(RL_cul_TP3_all+RL_cul_FN3_all)*100.0
        else:
          c3_RL_Recall_all = 0
        if(c3_RL_pricision_all+c3_RL_Recall_all!= 0):
           c3_RL_F1Score_all = 2*c3_RL_pricision_all*c3_RL_Recall_all/(c3_RL_pricision_all+c3_RL_Recall_all)
        else:
            c3_RL_F1Score_all = 0
        Avg_RL = (c1_RL_F1Score+c2_RL_F1Score+c3_RL_F1Score)/3
        Avg_RL_all = (c1_RL_F1Score_all+c2_RL_F1Score_all+c3_RL_F1Score_all)/3
        print("individual fscores are dqn {} {} {}".format(c1_RL_F1Score,c2_RL_F1Score,c3_RL_F1Score))
        print("individual fscores are  all {} {} {}".format(c1_RL_F1Score_all,c2_RL_F1Score_all,c3_RL_F1Score_all))
   

        print('RL accuracies are',c1_RL_F1Score,c2_RL_F1Score,c3_RL_F1Score)

        print("average  fsocre for baseline is {} and average fscore for RL is {} and two colab RL accuracy {}".format(Avg_baseline,Avg_RL,Avg_RL_all)) 
        total_cam1_occlusion = sum(dic_cam1_base_occlusion.values())
        total_cam2_occlusion = sum(dic_cam2_base_occlusion.values())
        total_cam3_occlusion = sum(dic_cam3_base_occlusion.values())
        average_occlusion_base = (total_cam1_occlusion+total_cam2_occlusion+total_cam3_occlusion)/3

        

        total_cam1_occlusion = sum(dic_cam1_DQN_occlusion.values())
        total_cam2_occlusion = sum(dic_cam2_DQN_occlusion.values())
        total_cam3_occlusion = sum(dic_cam3_DQN_occlusion.values())
        average_occlusion_DQN = (total_cam1_occlusion+total_cam2_occlusion+total_cam3_occlusion)/3
        #print(dic_cam_baseline_coverage.values())
        average_baseline_coverage = sum(dic_cam_baseline_coverage.values())/len(dic_cam_baseline_coverage.values())
        average_dqn_coverage = sum(dic_cam_dqn_coverage.values())/len(dic_cam_dqn_coverage.values())
        print("baseline occulution value is ",average_occlusion_base)
        print("RL occulusion value is ",average_occlusion_DQN)
        print("baseline coverage value is ",average_baseline_coverage)
        print("RL coverage value is ",average_dqn_coverage)
        
        ##write dic_cam1_angle_variation to file
        #save_name= str(init_angle_THETA1)+"_"+str(init_angle_THETA2)+"_"+str(init_angle_THETA3)
        #file_path = "./angle_var/"+str(save_name)+".txt"
        #with open(file_path, "w") as file:
        #  json.dump(dic_cams_angle_variation, file)


        #self.plot_occlusion(dic_cam1_base_occlusion,dic_cam2_base_occlusion,dic_cam3_base_occlusion,iteration_number,type_im="./plots/base_occlusion_"+str(iteration_number))
        #self.plot_occlusion(dic_cam1_DQN_occlusion,dic_cam2_DQN_occlusion,dic_cam3_DQN_occlusion,iteration_number,type_im="./plots/DQN_occlusion_"+str(iteration_number))
        #self.plot_able_variation(dic_cam1_angle_variation,dic_cam2_angle_variation,dic_cam3_angle_variation,iteration_number,type_im="./plots/angle_variation_"+str(iteration_number))
        #self.plot_actions(dic_test_summery_action,iteration_number,"Test")
        #self.plot_accuracy_per_frame(dic_cam1_baseline_accuracy,dic_cam2_baseline_accuracy,dic_cam3_baseline_accuracy,dic_cam1_DQN_accuracy,dic_cam2_DQN_accuracy,dic_cam3_DQN_accuracy,iteration_number)
        #self.plot_avg_accuracy_per_frame(dic_cam1_baseline_accuracy,dic_cam2_baseline_accuracy,dic_cam3_baseline_accuracy,dic_cam1_DQN_accuracy,dic_cam2_DQN_accuracy,dic_cam3_DQN_accuracy,iteration_number)
        #self.plot_able_variation2(dic_AB_overlap_variation,dic_AC_overlap_variation,dic_BC_overlap_variation)
        brake_occured=0
        print("rrrrrrrrrrrrrrrrrrr",average_baseline_coverage,average_dqn_coverage)
        if(len(dic_outofbound.keys()) < 0):
            global_test_summery[str(init_angle)] = [0,0,0,0,0,0]
        else:
            if(brake_occured == 0):
              global_test_summery_ABC[str(init_angle)] =[c1_baseline_F1Score,c2_baseline_F1Score,c3_baseline_F1Score,c1_RL_F1Score,c2_RL_F1Score,c3_RL_F1Score]
              global_test_summery[str(init_angle)] = [Avg_baseline,Avg_RL,average_occlusion_base,average_occlusion_DQN,average_baseline_coverage,average_dqn_coverage]
              global_yesSteeringYesComAI_summery[str(init_angle)] =[Avg_RL]
              global_NoSteeringNoComAI_summery[str(init_angle)]   =[Avg_baseline]
              global_NoSteeringYesComAI_summery[str(init_angle)]  =[Avg_RL_all]
              global_percentageOfcollaboration[str(init_angle)]   =[cummulative_AB,cummulative_BC,cummulative_AC]
              global_global_totaldetection_summary[str(init_angle)]   =[cummulative_detected_perframe_base/5000,cummulative_detected_perframe_dqn/5000]
              # these dictonaries contains the individual average accuracies of cameras calcukated over 500 frmaes.
              global_CAMA_Fscore[str(init_angle)] =[c1_baseline_F1Score,c1_RL_F1Score]
              global_CAMB_Fscore[str(init_angle)] =[c2_baseline_F1Score,c2_RL_F1Score]
              global_CAMC_Fscore[str(init_angle)] =[c3_baseline_F1Score,c3_RL_F1Score]
              print("iiiiiiiiiiiiiiiiiiiiiiiiiiii",T,global_global_totaldetection_summary)

        


        #logger.info("average  fsocre for baseline is: %s", Avg_baseline)
        #logger.info("average fscore for RL %s", Avg_RL)
        #logger.info("dic_outofbound %s", dic_outofbound)
        
       # logger.info("_________________________________________________________________________________________________________")
    


#  def render(self, mode='human'):
#    ...
#  def close (self):
#    ...
#env = gym.make("CartPole-v1")
#
#
#
#model = PPO("MlpPolicy", env, verbose=1)
#model.learn(total_timesteps=10_000)
#
#vec_env = model.get_env()
#for i in range(1000):
#    obs = vec_env.reset()
#    action, _states = model.predict(obs, deterministic=True)
#    obs, reward, done, info = vec_env.step(action)
#    vec_env.render()
#    print("steppppppp",i)
#    time.sleep(1)
#    # VecEnv resets automatically
#    # if done:
#    #   obs = env.reset()
#
#env.close()
def plot_actions_train(dic_action,iteration_number,type_im):
        # Plot all three sets of data in the same graph
         #plt.bar(x, y)
         plt.bar(list(dic_action.keys()), list(dic_action.values()), label="Actions")
         plt.xlabel("X-axis")
         plt.ylabel("Y-axis")
         plt.title("action_explored")
         plt.legend()
         plt.xticks(list(dic_action.keys()))
         plt.xticks(rotation=45)
         # Save the plot as an image file
         plt.savefig("./action_exploredTrained_"+type_im+"_"+str(iteration_number)+".png")
         plt.clf()
         # Show the plot
         #plt.show()
         return 0

def eval_weights_plot(model,env):
  weight_file_path="./best_model_4sofar/"
  # Use glob to get a list of all files in the directory
  file_list = glob.glob(weight_file_path + "*")
  
  # Iterate over the file list and load each file
  folder_path = "./video/"
  num_files_to_create = 5  # Change this to the desired number of files to create
  # Clear all folders in the location
  if os.path.exists(folder_path):
      # Remove all files and folders inside the folder
      shutil.rmtree(folder_path)
      print("Folders cleared successfully!")
  else:
      print("Folder does not exist.")

  for file_path in file_list:
    iteration_number=file_path.split("/")[-1].split("_")[3]
    print("ooooooooooooooooooooooooo",file_path)
    with open(file_path, "rb") as file:
        # Process or use the file as needed
        # For example, you can read the file contents or perform some operations
        # Example: Print the file path
        print("Loaded file:", file_path)
        model.set_parameters(file_path)
        model.set_env(env)
        env.startGame(model,iteration_number) 
        # Specify the folder containing the images
        image_folder = './video/'+str(iteration_number)+'/'
        # Specify the output video name
        video_name = './out_video/output_'+str(iteration_number)+'.mp4'
        # Specify the frames per second (fps) for the output video
        fps = 30
        # Call the function to convert the images to video
        convert_images_to_video(image_folder, video_name, fps)


def convert_images_to_video(image_folder, video_name, fps):
  
    images = sorted(os.listdir(image_folder))  # Get the list of image files
    number_images =len(images)
    if(number_images >100):
        image_path = os.path.join(image_folder, images[0])
        frame = cv2.imread(image_path)
        height, width, _ = frame.shape
        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for image in range(number_images):
            image_name= str(image)+".jpg"
            image_path = os.path.join(image_folder, image_name)
            frame = cv2.imread(image_path)
            video.write(frame)
        video.release()
def plot_initial_angle_summury(dic_action,iteration_number):
    # Plot all three sets of data in the same graph
     #plt.bar(x, y)
     plt5.figure(figsize=(200, 10)) 
     plt5.margins(x=0)
     plt5.bar(list(dic_action.keys()), list(dic_action.values()), label="Initial Angle")
     plt5.xlabel("angles")
     plt5.ylabel("Number of times angles explored")
     plt5.title("Initial Angle")
     plt5.legend()
     plt5.xticks(list(dic_action.keys()))
     plt5.xticks(rotation=90)
     
     # Save the plot as an image file
     plt5.savefig("./initAngle_explored_"+str(iteration_number)+".png")
     plt5.clf()
     # Show the plot
     #plt.show()
     return 0       


def plot_initial_angle_vs_acu_occu_rl_baseline(dic_action,iteration_number,hh):
    # Plot all three sets of data in the same graph
     #plt.bar(x, y)
     data=list(dic_action.values())
     base_acc = [row[0] for row in data]
     dqn_acc = [row[1] for row in data]
     base_occ = [row[2] for row in data]
     dqn_occ = [row[3] for row in data]
     base_cover = [row[4] for row in data]
     dqn_cover = [row[5] for row in data]
     

     plt5.figure(figsize=(200, 10)) 
     plt5.margins(x=0)
     plt5.plot(list(dic_action.keys()), base_acc, label="base_acc ")
     plt5.plot(list(dic_action.keys()), dqn_acc, label="dqn_acc ")
     plt5.xlabel("angles")
     plt5.ylabel("Number of times angles explored")
     plt5.title("Initial Angle")
     plt5.legend()
     plt5.xticks(list(dic_action.keys()))
     plt5.xticks(rotation=90)
     
     # Save the plot as an image file
     plt5.savefig("./model"+hh+"NewinitAngle_vs_rl_baseline_acc"+data_suppix+str(iteration_number)+".png")
     plt5.clf()
     
     plt5.figure(figsize=(50, 10)) 
     plt5.margins(x=0)
     plt5.bar(list(dic_action.keys()), base_occ, label="base_occ ")
     plt5.bar(list(dic_action.keys()), dqn_occ, label="dqn_occ ")
     plt5.xlabel("angles")
     plt5.ylabel("Averge Occlution of three frames")
     plt5.title("Initial Angle")
     plt5.legend()
     plt5.xticks(list(dic_action.keys()))
     plt5.xticks(rotation=90)
     plt5.savefig("./model1NewinitAngle_vs_rl_baseline_occ"+data_suppix+str(iteration_number)+".png")
     plt5.clf()


     plt5.figure(figsize=(200, 10)) 
     plt5.margins(x=0)
     plt5.plot(list(dic_action.keys()), base_cover, label="base_cover ")
     plt5.plot(list(dic_action.keys()), dqn_cover, label="dqn_cover")
     plt5.xlabel("angles")
     plt5.ylabel("Number of times angles explored")
     plt5.title("Initial Angle")
     plt5.legend()
     plt5.xticks(list(dic_action.keys()))
     plt5.xticks(rotation=90)
     plt5.savefig("./model1NewtrinminitAngle_vs_rl_baseline_cover"+data_suppix+str(iteration_number)+".png")
     plt5.clf()
     # Show the plot
     #plt.show()
     return 0 

def plot_final_plot(plot_type):
    # Set the path to the folder containing the images
    folder_path = './plots_collection/plots1/'
    
    # Define the name pattern to filter the images
    #image_pattern = 'action_explored_Test_*'
    image_pattern=plot_type
    
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    
    # Filter the files to only include those with the specified naming pattern
    selected_files = [file for file in files if file.startswith(image_pattern)]
    
    # Calculate the number of images and the number of rows and columns in the canvas
    num_images = len(selected_files)
    canvas_size = 4000
    canvas_size_height = 4000
    canvas_rows = int(np.sqrt(num_images))
    canvas_cols = int(np.ceil(num_images / canvas_rows))
    
    # Calculate the width and height of each cell in the canvas
    cell_width = int(canvas_size / canvas_cols)
    cell_height = int(canvas_size_height / canvas_rows)
    
    # Create an empty canvas to construct the final image
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    
    # Iterate over the selected files
    for i, file in enumerate(selected_files):
        # Load the image
        image_path = os.path.join(folder_path, file)
        image = cv2.imread(image_path)
    
        # Resize the image to fit within the cell
        resized_image = cv2.resize(image, (cell_width, cell_height))
    
        # Calculate the row and column indices of the current cell
        row = i // canvas_cols
        col = i % canvas_cols
    
        # Calculate the coordinates of the top-left corner of the current cell
        start_x = col * cell_width
        start_y = row * cell_height
    
        # Add the resized image to the canvas
        canvas[start_y:start_y + cell_height, start_x:start_x + cell_width] = resized_image
    
    # Display the constructed image
    #cv2.imshow("Constructed Image", canvas)
    #cv2.waitKey(0)
    
    # Write the constructed image to a location
    final_width=3000
    final_height=2000
    canvas = cv2.resize(canvas, (final_width, final_height))
    output_path = "./plots_collection/"+image_pattern+".jpg"
    cv2.imwrite(output_path, canvas)
    return 0
def read_dic_transformation():
      with open("./index_transformation_data.txt", "r") as file:
         lines = file.readlines()
      for line in lines:
         line = eval(line)
         dic_key= line.keys()
         str_dic_key=str(list(dic_key)[0])
         dic_a=line[str_dic_key]
         tf_dic[str_dic_key]=dic_a
read_dic_transformation()
def predict_ssd_model1(input_images):
  return ssd_model.predict(input_images)
# Define a function to predict using the second model
def predict_ssd_model2(input_images):
  return ssd_model2.predict(input_images)



def get_mAP_new(view= 1,image_id=0,det_content=[],A=""):
       mAP=10
       final_gt_path='./YOLOV3/'+dataset_load_name+'/'+A+"/"
       initialize_priors(view,final_gt_path)
       calc_accuracy(view,image_id,det_content,iou_tre=float(0.5),A=A)
       Precision,Recall,F1Score,TP,FP,FN,GT=get_Accuracy_Metrics()
       if(ENABLE_PRINT):
           print("Precision,Recall,F1Score,TP,FP,FN,GT",Precision,Recall,F1Score,TP,FP,FN,GT)
       return F1Score,TP,FP,FN


def looping_generate_load_gt():
  folder_path = "./YOLOV3/"+dataset_load_name  # Specify the path to the main folder
  subfolders = [os.path.basename(f.path) for f in os.scandir(folder_path) if f.is_dir()]
  
  for subfolder in subfolders:
      if(subfolder[0]=='A'):
       ff=1
      elif(subfolder[0]=='B'):
       ff=2
      elif(subfolder[0]=='C'):
        ff=3
      final_gt_path='./YOLOV3/'+dataset_load_name+'/'+subfolder+"/"
      initialize_priors(ff,final_gt_path)

    
     # break
def sub_detection_dictionary(file_mAP,file_TPFPFN,file_detection,view1= 1,view2= 2,view3= 3,value=0,ref='A10',colab1="B10",colab2="C10"):
  # in this method.given two views from two cameras. it should calculate the number of people in the view using yolo.and provide the value
  dic_col2_detection_results={}
  dic_col2_detection_results_TPFPFN={}
  dic_col2_detection_bbresults={}

  dic_col11_detection_results={}
  dic_col11_detection_results_TPFPFN={}
  dic_col11_detection_bbresults={}

  dic_col12_detection_results={}    
  dic_col12_detection_results_TPFPFN={} 
  dic_col12_detection_bbresults={}

  dic_Nocol_detection_results={}
  dic_Nocol_detection_results_TPFPFN={}
  dic_Nocol_detection_bbresults={}


  #start_time = time.time()    
  image_root_path='./YOLOV3/'+dataset_load_name+'/'
  foldername  = ref+'/'

  foldername_ref  = ref+'/'
  foldername_colab1  = colab1+'/'
  foldername_colab2  = colab2+'/'
  frameID=value
  if(len(str(frameID))==1):
   filename = "frame_000"+str(frameID)+".jpg"
  elif(len(str(frameID))==2):
   filename = "frame_00"+str(frameID)+".jpg"
  else:
   filename = "frame_0"+str(frameID)+".jpg"
  fnal_path_ref = image_root_path +'/' +foldername_ref + "View_00"+str(view1)+'/'
  fnal_path_colab1 = image_root_path +'/' +foldername_colab1 + "View_00"+str(view2)+'/'
  fnal_path_colab2 = image_root_path +'/' +foldername_colab2 + "View_00"+str(view3)+'/'
  image_cv=cv2.imread(fnal_path_ref+filename)
  image_out=cv2.resize(image_cv,(image_width,image_height))#image_cv.copy()
  #fps = (time.time() - start_time)*1000
  #print(f"time take for loading images : {fps:.2f} ms") 
  #image_root_path='./YOLOV3/'+dataset_load_name+'/' 
  #start_time = time.time()  
  input_images_ref = load_input_image(fnal_path_ref,value)
  
  
  input_images_colab1 = load_input_image(fnal_path_colab1,value)
  input_images_colab2 = load_input_image(fnal_path_colab2,value)
  cori_dic =tf_dic[ref+"_"+colab1]
  cori_dic2 =tf_dic[ref+"_"+colab2] # as of now i consider only one collaborator

  y_pred_colab = ssd_model.predict(input_images_colab1) # input is an image and output conf values 8732,21
  y_pred_colab2 = ssd_model.predict(input_images_colab2) # input is an image and output conf values 8732,21
  y_pred2_ref = ssd_model2.predict(input_images_ref) # input is an image and output conf values 8732,21
  y_pred2_ref_with2colab=y_pred2_ref.copy()
  y_pred2_ref_with1colab = y_pred2_ref.copy()
  y_pred2_ref_withNocolab = y_pred2_ref.copy()
  # Define a function to predict using the first model
  #pool = multiprocessing.Pool(processes=3)
  # Run the predictions in parallel
  #y_pred_colab1 = pool.apply_async(predict_ssd_model1, (input_images_colab1,))
  #y_pred_colab2 = pool.apply_async(predict_ssd_model1, (input_images_colab2,))
  #y_pred2_ref = pool.apply_async(predict_ssd_model2, (input_images_ref,))
  # Get the results
  #y_pred_colab = y_pred_colab1.get()
  #y_pred_colab2 = y_pred_colab2.get()
  #y_pred2_ref = y_pred2_ref.get()
  # Close the pool
  #pool.close()
  #pool.join()
  #print("y_pred_colab1",y_pred_colab.shape,y_pred_colab2.shape,y_pred2_ref.shape)
  y_pred4_ref = ssd_model4.predict(input_images_ref) # input is an image and out put is normal conf,bbobox
  #print(y_pred4_ref[0,1,:])
  y_pred5_colab = ssd_model5.predict(input_images_colab1) # input is an image and out put is normal conf,bbobox
  #y_pred2_ref=colab_process_singleCollaborator(y_pred2_ref,y_pred_colab,cori_dic)
  #fps = (time.time() - start_time)*1000
  #print(f"time take for process ssd : {fps:.2f} ms") 
  #start_time = time.time()
  ############################################################
  # this is for 2 colabs
  y_pred2_ref_2 = colab_process_twoCollaborator(y_pred2_ref_with2colab,y_pred_colab,y_pred_colab2,cori_dic,cori_dic2)
  #fps = (time.time() - start_time)*1000
  #print(f"time take for colab process: {fps:.2f} ms") 
  y_pred3 = ssd_model3.predict(y_pred2_ref_2)
  final=colab_concatanate_process(y_pred3,y_pred4_ref)
  det_indices_2colab=[]
  #tart_time = time.time() 
  output,det_indices = decode_y2(final,              # this is the ref cam after collabration
                          confidence_thresh=0.25,
                          iou_threshold=0.45,
                          top_k=20,
                          input_coords='centroids',
                          normalize_coords=True,
                          img_height=300,
                          img_width=300)
  for i in output[0]:
    if(len(i) != 0):
      det_indices_2colab.append([i[2],i[3],i[4],i[5]])
  mAP,TP,FP,FN = get_mAP_new(view1,frameID,det_indices_2colab,ref)
  #print('2 colab',mAP,TP,FP,FN)
  dic_index_2colab= ref+"_"+colab1+"_"+colab2 + "_View_00"+str(view1)+'_'+filename
  dic_col2_detection_results[dic_index_2colab] = mAP#det_content
  dic_col2_detection_bbresults[dic_index_2colab] = det_indices_2colab
  dic_col2_detection_results_TPFPFN[dic_index_2colab] = [TP,FP,FN]
  
  #2 colab results to be written to a file
  colab2_file_mAP=file_mAP[ref[0]+"_"+colab1[0]+"_"+colab2[0]]
  colab2_TPFPFN=file_TPFPFN[ref[0]+"_"+colab1[0]+"_"+colab2[0]]
  colab2_file_detection=file_detection[ref[0]+"_"+colab1[0]+"_"+colab2[0]]
  #print("writing  index is",colab2_file_mAP)
  for key, value in dic_col2_detection_results.items():
      colab2_file_mAP.write(f"{key}: {value}\n")
  for key, value in dic_col2_detection_results_TPFPFN.items():
      colab2_TPFPFN.write(f"{key}: {value}\n")
  for key, value in dic_col2_detection_bbresults.items():
      colab2_file_detection.write(f"{key}: {value}\n")
  ###############################################################
   ############################################################
  # this is for 1 colabs
  y_pred2_ref_1 = colab_process_singleCollaborator(y_pred2_ref_with1colab,y_pred_colab,cori_dic)
  #fps = (time.time() - start_time)*1000
  #print(f"time take for colab process: {fps:.2f} ms") 
  y_pred3 = ssd_model3.predict(y_pred2_ref_1)
  final=colab_concatanate_process(y_pred3,y_pred4_ref)
  det_indices_colab11=[]
  #tart_time = time.time() 
  output,det_indices = decode_y2(final,              # this is the ref cam after collabration
                          confidence_thresh=0.25,
                          iou_threshold=0.45,
                          top_k=20,
                          input_coords='centroids',
                          normalize_coords=True,
                          img_height=300,
                          img_width=300)
  for i in output[0]:
    if(len(i) != 0):
      det_indices_colab11.append([i[2],i[3],i[4],i[5]])
  mAP,TP,FP,FN = get_mAP_new(view1,frameID,det_indices_colab11,ref)
  #print('1 colab',mAP,TP,FP,FN)
  dic_index_11colab= ref+"_"+colab1+"_View_00"+str(view1)+'_'+filename
  dic_col11_detection_results[dic_index_11colab] = mAP#det_content
  dic_col11_detection_bbresults[dic_index_11colab] = det_indices_colab11
  dic_col11_detection_results_TPFPFN[dic_index_11colab] = [TP,FP,FN]
  
  #2 colab results to be written to a file
  colab11_file_mAP=file_mAP[ref[0]+"_"+colab1[0]]
  colab11_TPFPFN=file_TPFPFN[ref[0]+"_"+colab1[0]]
  colab11_file_detection=file_detection[ref[0]+"_"+colab1[0]]

  for key, value in dic_col11_detection_results.items():
      colab11_file_mAP.write(f"{key}: {value}\n")
  for key, value in dic_col11_detection_results_TPFPFN.items():
      colab11_TPFPFN.write(f"{key}: {value}\n")
  for key, value in dic_col11_detection_bbresults.items():
      colab11_file_detection.write(f"{key}: {value}\n")
  ###############################################################
   ############################################################
  # this is for 1 colabs
  y_pred2_ref_1 = colab_process_singleCollaborator(y_pred2_ref_with1colab,y_pred_colab2,cori_dic2)
  #fps = (time.time() - start_time)*1000
  #print(f"time take for colab process: {fps:.2f} ms") 
  y_pred3 = ssd_model3.predict(y_pred2_ref_1)
  final=colab_concatanate_process(y_pred3,y_pred4_ref)
  det_indices_colab12=[]
  #tart_time = time.time() 
  output,det_indices = decode_y2(final,              # this is the ref cam after collabration
                          confidence_thresh=0.25,
                          iou_threshold=0.45,
                          top_k=20,
                          input_coords='centroids',
                          normalize_coords=True,
                          img_height=300,
                          img_width=300)
  for i in output[0]:
    if(len(i) != 0):
      det_indices_colab12.append([i[2],i[3],i[4],i[5]])
  mAP,TP,FP,FN = get_mAP_new(view1,frameID,det_indices_colab12,ref)
  #print('one colab',mAP,TP,FP,FN)
  dic_index_11colab= ref+"_"+colab2+"_View_00"+str(view1)+'_'+filename
  dic_col12_detection_results[dic_index_11colab] = mAP#det_content
  dic_col12_detection_bbresults[dic_index_11colab] = det_indices_colab12
  dic_col12_detection_results_TPFPFN[dic_index_11colab] = [TP,FP,FN]
  
  #2 colab results to be written to a file
  colab12_file_mAP=file_mAP[ref[0]+"_"+colab2[0]]
  colab12_TPFPFN=file_TPFPFN[ref[0]+"_"+colab2[0]]
  colab12_file_detection=file_detection[ref[0]+"_"+colab2[0]]
  for key, value in dic_col12_detection_results.items():
      colab12_file_mAP.write(f"{key}: {value}\n")
  for key, value in dic_col12_detection_results_TPFPFN.items():
      colab12_TPFPFN.write(f"{key}: {value}\n")
  for key, value in dic_col12_detection_bbresults.items():
      colab12_file_detection.write(f"{key}: {value}\n")
  ###############################################################
   ############################################################
  # this is for No colaboration
  #fps = (time.time() - start_time)*1000
  #print(f"time take for colab process: {fps:.2f} ms") 
  y_pred2_ref_withNocolab =y_pred2_ref_withNocolab[np.newaxis, ...]
  y_pred3 = ssd_model3.predict(y_pred2_ref_withNocolab)
  final=colab_concatanate_process(y_pred3,y_pred4_ref)
  det_content_nocolab=[]
  #tart_time = time.time() 
  output,det_content = decode_y2(final,              # this is the ref cam after collabration
                          confidence_thresh=0.25,
                          iou_threshold=0.45,
                          top_k=20,
                          input_coords='centroids',
                          normalize_coords=True,
                          img_height=300,
                          img_width=300)
  for i in output[0]:
    if(len(i) != 0):
      det_content_nocolab.append([i[2],i[3],i[4],i[5]])
  dic_index_Nocolab= ref+"_View_00"+str(view1)+'_'+filename
  mAP,TP,FP,FN = get_mAP_new(view1,frameID,det_content_nocolab,ref)
  #print('No colab',mAP,TP,FP,FN)
  dic_Nocol_detection_results[dic_index_Nocolab] = mAP#det_content
  dic_Nocol_detection_bbresults[dic_index_Nocolab] = det_content_nocolab
  dic_Nocol_detection_results_TPFPFN[dic_index_Nocolab] = [TP,FP,FN]
  
  #2 colab results to be written to a file
  Nocolab_file_mAP=file_mAP[ref[0]]
  Nocolab_TPFPFN=file_TPFPFN[ref[0]]
  Nocolab_file_detection=file_detection[ref[0]]
  for key, value in dic_Nocol_detection_results.items():
      Nocolab_file_mAP.write(f"{key}: {value}\n")
  for key, value in dic_Nocol_detection_results_TPFPFN.items():
      Nocolab_TPFPFN.write(f"{key}: {value}\n")
  for key, value in dic_Nocol_detection_bbresults.items():
      Nocolab_file_detection.write(f"{key}: {value}\n")
  ###############################################################
  #fps = (time.time() - start_time)*1000
  #print(f"time take for decode output: {fps:.2f} ms") 
  #fps = (time.time() - start_time)*1000
  #print(f"time take for comAI process: {fps:.2f} ms") 
  #print(output)
  # this has to changeeeeeeeee
  #image,det_content = yolo.detect_image(image,image_cv,ffx,value)
  #start_time = time.time() 
  people = len(det_content)
  #fps = (time.time() - start_time)*1000
  #print(f"time take for file write process: {fps:.2f} ms") 
  #print(f"*******************************************FPS: {fps:.2f} ms")
  return people
def looping_generate_detection_dictionary():
  folder_path = "./YOLOV3/"+dataset_load_name  # Specify the path to the main folder
  subfolders = [os.path.basename(f.path) for f in os.scandir(folder_path) if f.is_dir()]
  #print("number of tuples ref,col1,col2 are",len(total_collaboration_tuples))
  #with open("./all_detection_mAP_results_"+data_suppix+"comAI.txt", "a") as file_mAP:
  #  print("loaded"+"./all_detection_mAP_results_"+data_suppix+"comAI.txt")
  #  with open("./all_detection_TPFPFN_results_"+data_suppix+"comAI.txt", "a") as file_TPFPFN:
  #    print("loaded"+"./all_detection_TPFPFN_results_"+data_suppix+"comAI.txt")
  #    with open("./all_detection_results"+data_suppix+"comAI.txt", "a") as file_detection:
  #      print("loaded"+"./all_detection_results"+data_suppix+"comAI.txt")
  # Assuming you have a list of data_suppix values
  data_suppix_list = ["A_B_C", "A_B","A_C", "A","B_A_C","B_A","B_C","B","C_A_B","C_A","C_B", "C"]
  file_mAP={}
  file_TPFPFN={}
  file_detection={}
  inn=0
  for data_suppix in data_suppix_list:
  
      file_mAP_1= open("./out_test/new_detection_mAP_results_"+data_suppix+"comAI.txt", "a")
      file_mAP[data_suppix]=file_mAP_1
      print("loaded"+".all_detection_mAP_results_"+data_suppix+"comAI.txt")
      file_TPFPFN_1=  open("./out_test/new_detection_TPFPFN_results_"+data_suppix+"comAI.txt", "a")
      file_TPFPFN[data_suppix]=file_TPFPFN_1
      print("loaded"+"./all_detection_TPFPFN_results_"+data_suppix+"comAI.txt")
      file_detection_1 = open("./out_test/new_detection_results"+data_suppix+"comAI.txt", "a") 
      file_detection[data_suppix]=file_detection_1
      print("loaded"+"./all_detection_results"+data_suppix+"comAI.txt")
      inn+=1
  for i in tqdm(range(len(total_collaboration_tuples))):
  #for subfolder in total_collaboration_tuples:
      subfolder=total_collaboration_tuples[i]
      print(subfolder)
      if(subfolder[0][0]=='A'):
       ff1=1
      elif(subfolder[0][0]=='B'):
       ff1=2
      elif(subfolder[0][0]=='C'):
       ff1=3
      if(subfolder[1][0]=='A'):
       ff2=1
      elif(subfolder[1][0]=='B'):
       ff2=2
      elif(subfolder[1][0]=='C'):
       ff2=3
      if(subfolder[2][0]=='A'):
       ff3=1
      elif(subfolder[2][0]=='B'):
       ff3=2
      elif(subfolder[2][0]=='C'):
       ff3=3
      start_time = time.time()
      # Create a ThreadPoolExecutor with 3 threads
      # Define a function for the first loop
      for value in range(500):
        #start_time = time.time()
        sub_detection_dictionary(file_mAP,file_TPFPFN,file_detection,view1= ff1,view2= ff2,view3= ff3,value=value,ref=subfolder[0],colab1=subfolder[1],colab2=subfolder[2])   
      fps = (time.time() - start_time)*1000
      print(f"time take for comple the loop is : {fps:.2f} ms") 
      #break
combinations_new=[(0,0,0),(10,10,10),(20,20,20),(30,30,30),(40,40,40),(50,50,50),(60,60,60),(70,70,70),(80,80,80),(90,90,90)]
def finding_lowest_fscore_combinarion(yolo):  #combinations
      #dic_test_summery_action={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0,19:0,20:0,21:0,22:0,23:0,24:0,25:0,26:0}
      dic_test_summery_action={0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0, 39: 0, 40: 0, 41: 0, 42: 0, 43: 0, 44: 0, 45: 0, 46: 0, 47: 0, 48: 0, 49: 0, 50: 0, 51: 0, 52: 0, 53: 0, 54: 0, 55: 0, 56: 0, 57: 0, 58: 0, 59: 0, 60: 0, 61: 0, 62: 0, 63: 0, 64: 0, 65: 0, 66: 0, 67: 0, 68: 0, 69: 0, 70: 0, 71: 0, 72: 0, 73: 0, 74: 0, 75: 0, 76: 0, 77: 0, 78: 0, 79: 0, 80: 0, 81: 0, 82: 0, 83: 0, 84: 0, 85: 0, 86: 0, 87: 0, 88: 0, 89: 0, 90: 0, 91: 0, 92: 0, 93: 0, 94: 0, 95: 0, 96: 0, 97: 0, 98: 0, 99: 0, 100: 0, 101: 0, 102: 0, 103: 0, 104: 0, 105: 0, 106: 0, 107: 0, 108: 0, 109: 0, 110: 0, 111: 0, 112: 0, 113: 0, 114: 0, 115: 0, 116: 0, 117: 0, 118: 0, 119: 0, 120: 0, 121: 0, 122: 0, 123: 0, 124: 0, 125: 0, 126: 0, 127: 0, 128: 0, 129: 0, 130: 0, 131: 0, 132: 0, 133: 0, 134: 0, 135: 0, 136: 0, 137: 0, 138: 0, 139: 0, 140: 0, 141: 0, 142: 0, 143: 0, 144: 0, 145: 0, 146: 0, 147: 0, 148: 0, 149: 0, 150: 0, 151: 0, 152: 0, 153: 0, 154: 0, 155: 0, 156: 0, 157: 0, 158: 0, 159: 0, 160: 0, 161: 0, 162: 0, 163: 0, 164: 0, 165: 0, 166: 0, 167: 0, 168: 0, 169: 0, 170: 0, 171: 0, 172: 0, 173: 0, 174: 0, 175: 0, 176: 0, 177: 0, 178: 0, 179: 0, 180: 0, 181: 0, 182: 0, 183: 0, 184: 0, 185: 0, 186: 0, 187: 0, 188: 0, 189: 0, 190: 0, 191: 0, 192: 0, 193: 0, 194: 0, 195: 0, 196: 0, 197: 0, 198: 0, 199: 0, 200: 0, 201: 0, 202: 0, 203: 0, 204: 0, 205: 0, 206: 0, 207: 0, 208: 0, 209: 0, 210: 0, 211: 0, 212: 0, 213: 0, 214: 0, 215: 0}

      dic_summary={}
      dic_summary_TP={}
      dic_summary_FP={}
      dic_summary_TN={}

      for combination in combinations_new: #if AI     
        THETA1_LOWER_BOUND_ANGLE = combination[0]
        THETA2_LOWER_BOUND_ANGLE = combination[1]
        THETA3_LOWER_BOUND_ANGLE = combination[2]
        cul_TP1,cul_FP1,cul_FN1 =0,0,0
        cul_TP2,cul_FP2,cul_FN2 =0,0,0
        cul_TP3,cul_FP3,cul_FN3 =0,0,0

        dic_cam1_angle_variation={}
        dic_cam2_angle_variation={}
        dic_cam3_angle_variation={}
        dic_cam1_baseline_accuracy={}
        dic_cam2_baseline_accuracy={}
        dic_cam3_baseline_accuracy={}

        
        print("rrrrrrrrrrrrr",THETA1_LOWER_BOUND_ANGLE,THETA2_LOWER_BOUND_ANGLE,THETA3_LOWER_BOUND_ANGLE)
        for video_frame in range (0,500):
          print(video_frame)
          base_line_THETA1= THETA1_LOWER_BOUND_ANGLE
          base_line_THETA2= THETA2_LOWER_BOUND_ANGLE
          base_line_THETA3= THETA3_LOWER_BOUND_ANGLE
          A_x = "A"+str(base_line_THETA1)
          B_x = "B"+str(base_line_THETA2)
          C_x = "C"+str(base_line_THETA3)
          yy1,det_content1 = get_people(view1= 1,view2= 2,view3= 3,value=video_frame,ref=A_x,colab1=B_x,colab2=C_x,colab_action=[0,0,0])
          yy2,det_content2 = get_people(view1= 2,view2= 1,view3= 3,value=video_frame,ref=B_x,colab1=A_x,colab2=C_x,colab_action=[0,0,0])
          yy3,det_content3 = get_people(view1= 3,view2= 1,view3= 2,value=video_frame,ref=C_x,colab1=A_x,colab2=B_x,colab_action=[0,0,0])


          TP1,FP1,FN1= get_TPFPFN(view =1,image_id=video_frame,det_content=det_content1,theta1 = base_line_THETA1 ,theta2 = base_line_THETA2,theta3 = base_line_THETA3,ref=A_x,colab1=B_x,colab2=C_x)
          TP2,FP2,FN2= get_TPFPFN(view =2,image_id=video_frame,det_content=det_content2,theta1 = base_line_THETA1 ,theta2 = base_line_THETA2,theta3 = base_line_THETA3,ref=B_x,colab1=A_x,colab2=C_x)
          TP3,FP3,FN3= get_TPFPFN(view =3,image_id=video_frame,det_content=det_content3,theta1 = base_line_THETA1 ,theta2 = base_line_THETA2,theta3 = base_line_THETA3,ref=C_x,colab1=A_x,colab2=B_x)
  
          cul_TP1+= TP1
          cul_FP1+= FP1
          cul_FN1+= FN1
          cul_TP2+= TP2
          cul_FP2+= FP2
          cul_FN2+= FN2
          cul_TP3+= TP3
          cul_FP3+= FP3
          cul_FN3+= FN3
          if(TP1+FP1 != 0):
            current_c1_baseline_pricision = float(TP1)/(TP1+FP1)*100.0
          else:
            current_c1_baseline_pricision = 0
          if(TP2+FP2 != 0):
            current_c2_baseline_pricision = float(TP2)/(TP2+FP2)*100.0
          else:
            current_c2_baseline_pricision = 0
          if(TP3+FP3 != 0):
            current_c3_baseline_pricision = float(TP3)/(TP3+FP3)*100.0
          else:
            current_c3_baseline_pricision = 0

          if(TP1+FN1 != 0):
            current_c1_baseline_recall = float(TP1)/(TP1+FN1)*100.0
          else:
            current_c1_baseline_recall = 0
          if(TP2+FN2 != 0):
            current_c2_baseline_recall = float(TP2)/(TP2+FN2)*100.0
          else:
            current_c2_baseline_recall = 0
          if(TP3+FN3 != 0):
            current_c3_baseline_recall = float(TP3)/(TP3+FN3)*100.0
          else:
            current_c3_baseline_recall = 0


          if(current_c1_baseline_pricision+current_c1_baseline_recall):
            current_c1_baseline_F1Score = 2*current_c1_baseline_pricision*current_c1_baseline_recall/(current_c1_baseline_pricision+current_c1_baseline_recall)
          else:
            current_c1_baseline_F1Score = 0

          if(current_c2_baseline_pricision+current_c2_baseline_recall):
            current_c2_baseline_F1Score = 2*current_c2_baseline_pricision*current_c2_baseline_recall/(current_c2_baseline_pricision+current_c2_baseline_recall)
          else:
            current_c2_baseline_F1Score = 0

          if(current_c3_baseline_pricision+current_c3_baseline_recall):
            current_c3_baseline_F1Score = 2*current_c3_baseline_pricision*current_c3_baseline_recall/(current_c3_baseline_pricision+current_c3_baseline_recall)
          else:
            current_c3_baseline_F1Score = 0

          dic_cam1_baseline_accuracy[video_frame] = [current_c1_baseline_F1Score]
          dic_cam2_baseline_accuracy[video_frame] = [current_c2_baseline_F1Score]
          dic_cam3_baseline_accuracy[video_frame] = [current_c3_baseline_F1Score]

        
        
        dic_summary_TP[A_x] = cul_TP1
        dic_summary_FP[A_x] = cul_FP1
        dic_summary_TN[A_x] = cul_FN1
        dic_summary_TP[B_x] = cul_TP2
        dic_summary_FP[B_x] = cul_FP2
        dic_summary_TN[B_x] = cul_FN2
        dic_summary_TP[C_x] = cul_TP3
        dic_summary_FP[C_x] = cul_FP3
        dic_summary_TN[C_x] = cul_FN3
        print(dic_summary_TP)
        print(dic_summary_FP)
        print(dic_summary_TN)
      for combination in combinations:
        new_angle1 = "A"+str(combination[0])
        new_angle2 = "B"+str(combination[1])
        new_angle3 = "C"+str(combination[2])
        cul_TP1 = dic_summary_TP[new_angle1]
        cul_FP1 = dic_summary_FP[new_angle1]
        cul_FN1 = dic_summary_TN[new_angle1]
        cul_TP2 = dic_summary_TP[new_angle2]
        cul_FP2 = dic_summary_FP[new_angle2]
        cul_FN2 = dic_summary_TN[new_angle2] 
        cul_TP3 = dic_summary_TP[new_angle3] 
        cul_FP3 = dic_summary_FP[new_angle3] 
        cul_FN3 = dic_summary_TN[new_angle3]
        c1_baseline_pricision = float(cul_TP1)/(cul_TP1+cul_FP1)*100.0   
        c1_baseline_Recall = float(cul_TP1)/(cul_TP1+cul_FN1)*100.0
        c1_baseline_F1Score = 2*c1_baseline_pricision*c1_baseline_Recall/(c1_baseline_pricision+c1_baseline_Recall)
        c2_baseline_pricision = float(cul_TP2)/(cul_TP2+cul_FP2)*100.0   
        c2_baseline_Recall = float(cul_TP2)/(cul_TP2+cul_FN2)*100.0
        c2_baseline_F1Score = 2*c2_baseline_pricision*c2_baseline_Recall/(c2_baseline_pricision+c2_baseline_Recall)
        c3_baseline_pricision = float(cul_TP3)/(cul_TP3+cul_FP3)*100.0   
        c3_baseline_Recall = float(cul_TP3)/(cul_TP3+cul_FN3)*100.0
        c3_baseline_F1Score = 2*c3_baseline_pricision*c3_baseline_Recall/(c3_baseline_pricision+c3_baseline_Recall)
        Avg_baseline = (c1_baseline_F1Score+c2_baseline_F1Score+c3_baseline_F1Score)/3
        dic_summary[combination] = Avg_baseline
      
      print('minimum value of the dic is',min(dic_summary.values()))
      # Create a new OrderedDict based on the accessing order of values
      access_ordered_dict = OrderedDict(sorted(dic_summary.items(), key=lambda x: x[1]))
      print(access_ordered_dict)
      # Write the dictionary to a text file
      with open("final_log_init_angle_accuracy.txt", "w") as file:
          for key, value in access_ordered_dict.items():
              file.write(f"{key}: {value}\n")
        

def get_TPFPFN(self,view= 1,image_id=0,det_content=[],theta1=0,theta2=0,theta3=0,ref="A10",colab1="B10",colab2="C10"):
  mAP=10
  #foldername  = 'A'+str(theta1)+'B'+str(theta2)+'C'+str(theta3)+'/'
  #final_gt_path='./YOLOV3/final_sim_images/'+A+"/"
  #initialize_priors(view,final_gt_path)
  #calc_accuracy(view,image_id,det_content,iou_tre=float(0.5),A=A)
  #Precision,Recall,F1Score,TP,FP,FN,GT=get_Accuracy_Metrics()
  value=image_id
  if(len(str(value))==1):
      imageName =  "frame_000"+str(value)+".jpg"
  elif(len(str(value))==2):
      imageName = "frame_00"+str(value)+".jpg"
  else:
      imageName = "frame_0"+str(value)+".jpg"
  self.en_from_text = 1
  if(self.en_from_text == 0):
     final_gt_path='./YOLOV3/'+dataset_load_name+'/'+A+"/"
     initialize_priors(view,final_gt_path)
     calc_accuracy(view,image_id,det_content,iou_tre=float(0.5),A=A)
     Precision,Recall,F1Score,TP,FP,FN,GT=get_Accuracy_Metrics()
  else:
    #this part has to be changed accordingly. consider a camera A. at a given moment there are 3 possible views. there are 4 possible instances  for camera A.
    # A without collaboration, A with B, A with C. and A with B and C. so there are 4 possible instances.
    AB=colab_action[0]
    AC=colab_action[1]
    BC=colab_action[2]
    dic_detection_results_index = ""
    index_dic=""
    if(ref[0]=="A"):
      #if(AB==1 and AC==1):
      dic_detection_results_index += "A"
      index_dic += ref
      if(AB==1):
         dic_detection_results_index += "_B"
         index_dic += "_"+colab1
      if(AC==1):  
        dic_detection_results_index += "_C"
        index_dic += "_"+colab2
    elif(ref[0]=="B"):
      dic_detection_results_index += "B"
      index_dic += ref
      if(AB==1):
         dic_detection_results_index += "_A"
         index_dic += "_"+colab1
      if(BC==1):    
        dic_detection_results_index += "_C"
        index_dic += "_"+colab2
    elif(ref[0]=="C"):
      dic_detection_results_index += "C"
      index_dic += ref
      if(AC==1):  
        dic_detection_results_index += "_A"
        index_dic += "_"+colab1
      if(BC==1):    
        dic_detection_results_index += "_B"
        index_dic += "_"+colab2
    temp_dic=dic_TPFPFN_results[dic_detection_results_index]
    if(len(str(frameID))==1):
      filename =  "frame_000"+str(frameID)+".jpg"
    elif(len(str(frameID))==2):
     filename = "frame_00"+str(frameID)+".jpg"
    else:
     filename =  "frame_0"+str(frameID)+".jpg"
    index_dic +=  "_View_00"+str(view)+'_'+filename
    TP,FP,FN= temp_dic[index_dic] 
  if(ENABLE_PRINT):
    print("Precision,Recall,F1Score,TP,FP,FN,GT",Precision,Recall,F1Score,TP,FP,FN,GT)
  return TP,FP,FN

def get_people(self,view1= 1,view2=2,view3=3,value=0,ref='A10',colab1='B10',colab2='C10',colab_action=[]):
  self.en_from_text = 1
  frameID=value
  if(self.en_from_text == 0):
    # in this method.given two views from two cameras. it should calculate the number of people in the view using yolo.and provide the value
    image_root_path='./YOLOV3/'+dataset_load_name
    
    foldername_ref  = ref+'/'
    foldername_colab1  = colab1+'/'
    foldername_colab2  = colab2+'/'
    fnal_path_ref = image_root_path +'/' +foldername_ref+"View_00"+str(view1)+'/' 
    fnal_path_colab1 = image_root_path +'/' +foldername_colab1+"View_00"+str(view2)+'/'
    fnal_path_colab2 = image_root_path +'/' +foldername_colab2+"View_00"+str(view3)+'/'
    input_images_ref = load_input_image(fnal_path_ref,value)
    input_images_colab1 = load_input_image(fnal_path_colab1,value)
    input_images_colab2 = load_input_image(fnal_path_colab2,value)
    #pdb.set_trace()
    #print(self.dic_transformation_state.keys())
    #print("pppppppppp",ref+"_"+colab1)
    cori_dic =self.dic_transformation_state[ref+"_"+colab1]
    cori_dic2 =self.dic_transformation_state[ref+"_"+colab2]
    #cori_dic2 =self.dic_transformation_state[ref+"_"+colab2] # as of now i consider only one collaborator
    
    y_pred_colab = ssd_model.predict(input_images_colab1) # input is an image and output conf values 8732,21
    y_pred_colab2 = ssd_model.predict(input_images_colab2) # input is an image and output conf values 8732,21
    y_pred2_ref = ssd_model2.predict(input_images_ref) # input is an image and output conf values 8732,21
    #print("inputtttttttttttttttttttttttttt",input_images2_ref[0,0,:,:])
    y_pred4_ref = ssd_model4.predict(input_images_ref) # input is an image and out put is normal conf,bbobox
    #print(y_pred4_ref[0,1,:])
    y_pred5_colab = ssd_model5.predict(input_images_colab1) # input is an image and out put is normal conf,bbobox
    #y_pred2_ref=colab_process_singleCollaborator(y_pred2_ref,y_pred_colab,cori_dic)
    y_pred2_ref = colab_process_twoCollaborator(y_pred2_ref,y_pred_colab,y_pred_colab2,cori_dic,cori_dic2)
    y_pred3 = ssd_model3.predict(y_pred2_ref)
    final=colab_concatanate_process(y_pred3,y_pred4_ref)
    det_content=[]
    output,det_indices = decode_y2(final,              # this is the ref cam after collabration
                            confidence_thresh=0.25,
                            iou_threshold=0.45,
                            top_k=10,
                            input_coords='centroids',
                            normalize_coords=True,
                            img_height=300,
                            img_width=300)
    #print(output)
    for i in output[0]:
      #print(i,len(output) )
      if(len(i) != 0):
        det_content.append([i[2],i[3],i[4],i[5]])
    image_out=input_images_ref
  else:
    #this part has to be changed accordingly. consider a camera A. at a given moment there are 3 possible views. there are 4 possible instances  for camera A.
    # A without collaboration, A with B, A with C. and A with B and C. so there are 4 possible instances.
    AB=colab_action[0]
    AC=colab_action[1]
    BC=colab_action[2]
    dic_detection_results_index = ""
    index_dic=""
    if(ref[0]=="A"):
      #if(AB==1 and AC==1):
      dic_detection_results_index += "A"
      index_dic += ref
      if(AB==1):
         dic_detection_results_index += "_B"
         index_dic += "_"+colab1
      if(AC==1):  
        dic_detection_results_index += "_C"
        index_dic += "_"+colab2
    elif(ref[0]=="B"):
      dic_detection_results_index += "B"
      index_dic += ref
      if(AB==1):
         dic_detection_results_index += "_A"
         index_dic += "_"+colab1
      if(BC==1):    
        dic_detection_results_index += "_C"
        index_dic += "_"+colab2
    elif(ref[0]=="C"):
      dic_detection_results_index += "C"
      index_dic += ref
      if(AC==1):  
        dic_detection_results_index += "_A"
        index_dic += "_"+colab1
      if(BC==1):    
        dic_detection_results_index += "_B"
        index_dic += "_"+colab2
    temp_dic=dic_detection_results[dic_detection_results_index]
    if(len(str(frameID))==1):
      filename =  "frame_000"+str(frameID)+".jpg"
    elif(len(str(frameID))==2):
     filename = "frame_00"+str(frameID)+".jpg"
    else:
     filename =  "frame_0"+str(frameID)+".jpg"
    index_dic +=  "_View_00"+str(view1)+'_'+filename
    det_content= temp_dic[index_dic] 
  fnal_path_ref = image_root_path +'/' +foldername_ref+"View_00"+str(view1)+'/' 
  image_out=load_input_image(fnal_path_ref,value)
  people = len(det_content)
  image_out=normalize_image(image_out)
  #print('ttttttttttttttttttttttttttttttttttt',det_content)
  return image_out,det_content

def compareModelWeights(model_a, model_b):
  for p1, p2 in zip(model_a,model_b):
        aa=model_a[p1]
        bb=model_b[p2]
        print(aa)
        print(bb)
        if torch.equal(aa, bb):
            pass
            print('pass')
        else:
           return False
  return True

def eval_best_weight_all_initangles(model,model2,env):
  iteration_number=5000000#
  hh="Tbest_model_test24best1"#best_model_test4" best_model_6
  weight_file_path="./"+hh+"/dqn_best_model_"+str(iteration_number)+"_steps" # #best_model_4sofar
  weight_file_path2="./best_model_6/dqn_best_model_"+str(3000000)+"_steps"
  weight_file_path="./Tbest_model_test24best1/best_model"# best_model_test9
  # Use glob to get a list of all files in the directory
  en_combination=0
  random.seed(42)
  if(en_combination == 1):
    for i in tqdm(range(len(combinations))):#tqdm(range(len(combinations))): #tqdm(range(100)):
    #for combination in combinations:
      i = random.randint(0, 999)
      combination = combinations[i]
      print("combination is:", combination)
      print("Loaded file:", weight_file_path)
      model.set_parameters(weight_file_path)
      model.set_env(env)
      env.startGame_test(model,iteration_number,combination,hh,10) 
  else:
    for i in range(10,11):
      for combination in [[30,40,80],[20,20,80],[0,90,0],[80,40,70]]: #[[30,40,80],[20,20,80],[0,90,0],[80,40,70],[40,10,0],[0,0,0],[60,10,90],[70,90,50],[80,90,0],[90,90,0]]:
        print("combination is:", combination)
        print("Loaded file:", weight_file_path)
        model.set_parameters(weight_file_path)

        model.set_env(env)
        env.startGame_test(model,iteration_number,combination,hh,i) 
    

  
  ## Write the dictionary to a text file
  #with open("./final_all_combinations_accuracy_and_occlusion.txt", "w") as file:
  #  json.dump(global_test_summery, file)
#
#  #with open("./final_all_combinations_accuracy_and_occlusionglobal_test_summery_ABC"+str(hh)+".txt", "w") as file:
#  #    for key, value in global_test_summery_ABC.items():
#  #        file.write(f"{key}: {value}\n")
#
#  plot_initial_angle_vs_acu_occu_rl_baseline(global_test_summery,iteration_number,hh)
#  file_path = "./base_accuracy.txt"
#  with open(file_path, "w") as file:
#    json.dump(global_NoSteeringNoComAI_summery, file)
#
#  file_path = "./Steer(Yes)_comAI(yes)_accuracy(ours).txt"
#  with open(file_path, "w") as file:
#    json.dump(global_yesSteeringYesComAI_summery, file)
  #file_path = "./Steer(No)_comAI(yes)_accuracy.txt"
  #with open(file_path, "w") as file:
  #  json.dump(global_yesSteeringYesComAI_summery, file)
#
#  file_path = "./Steer(yes)_comAI(yesALL)_accuracy.txt"
#  with open(file_path, "w") as file:
#    json.dump(global_NoSteeringYesComAI_summery, file)
#
#  file_path = "./percentageOfcollaboration.txt"
#  with open(file_path, "w") as file:
#    json.dump(global_percentageOfcollaboration, file)
#
#  #file_path = "./global_(onlycomAI)detection_number.txt"
#  #with open(file_path, "w") as file:
#  #  json.dump(global_global_totaldetection_summary, file)
#
#  #file_path = "./global_(onlySreeting)detection_number(ourpolicy).txt"
#  #with open(file_path, "w") as file:
#  #  json.dump(global_global_totaldetection_summary, file)
#  
#  file_path = "./global_detection_numberfmap_colabonly.txt"
#  with open(file_path, "w") as file:
#    json.dump(global_global_totaldetection_summary, file)
#
#  file_path = "./global_cameraA_f1score_comparison.txt"
#  with open(file_path, "w") as file:
#    json.dump(global_CAMA_Fscore, file)
#
#  file_path = "./global_cameraB_f1score_comparison.txt"
#  with open(file_path, "w") as file:
#    json.dump(global_CAMB_Fscore, file)
#
#  file_path = "./global_cameraC_f1score_comparison.txt"
#  with open(file_path, "w") as file:
#    json.dump(global_CAMC_Fscore, file)
#
def save_best_model_callback(eval_env):
    checkpoint_callback = CheckpointCallback(save_freq=5000000, save_path='./best_model_test17/',name_prefix='dqn_best_model')
    return checkpoint_callback

log_dirr="./best_model_test17"
def main(game):

# to test Q learning player
 if(game==1):

  env = CustomEnv(traning=False)
  print("rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")

  # If the environment don't follow the interface, an error will be thrown
  tensorboard_log = "./logsComAI3/"
  env = Monitor(env, log_dirr)
  model = DQN(
    "CnnPolicy",
    env,
    verbose=1,
    train_freq=(4, "step"), #16 (4, "step")
    gradient_steps= 1, #8
    gamma=1,
    exploration_fraction=0.5 , #0.375
    exploration_final_eps=0.01,
    target_update_interval=200, #600
    learning_starts=100000, #1000nvidia-smi
    buffer_size=100000, #200000
    batch_size=32,#,128,
    learning_rate=1e-7,
    tensorboard_log=tensorboard_log,#tensorboard_log
    seed=2, 
    device="cuda",
    policy_kwargs=dict(net_arch=[150, 150]), #dict(net_arch=[256, 256])
  )
  
#  start_time = time.time()
  # given a best weight file and test it on all the initial angles.plot the occlution and accuracy and coverage area.
  
  #eval_best_weight_all_initangles(model,model,env) 

  #best_model_6 is the new one with added last line to reward function
  callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir= log_dirr)
  eval_callback = EvalCallback(env, best_model_save_path=log_dirr,log_path=log_dirr, eval_freq=1000, deterministic=True, render=False)
  #callback_list = CallbackList([TensorboardCallback(env), save_best_model_callback(env)])
  callback_list = CallbackList([TensorboardCallback(env),eval_callback,save_best_model_callback(env)])
  total_timesteps=10
  #checkpoint_path="./best_model_test4/dqn_best_model_"+str(5000000)+"_steps"
  #checkpoint_path="./best_model_test4/best_model"
  #model.set_parameters(checkpoint_path)
  start_from_0=1
  if start_from_0==1:
     start_time = time.time()
     model.learn(total_timesteps=5000000, callback=callback_list,progress_bar=True,log_interval=64)#10000  1850000
     iteration_number=3
     plot_actions_train(dic_summery_action,iteration_number,"episode_lenth")
     plot_initial_angle_summury(dic_init_angle_summery,iteration_number)
     # Record the end time
     end_time = time.time()
     # Calculate the elapsed time
     elapsed_time = end_time - start_time
     # Convert elapsed time to a timedelta object
     delta = datetime.timedelta(seconds=elapsed_time)
     # Extract hours, minutes, and seconds from the timedelta object
     hours = delta.seconds // 3600
     minutes = (delta.seconds // 60) % 60
     seconds = delta.seconds % 60
     with open("./Train_action_execution"+str(iteration_number)+".txt", "a") as file:
        for key, value in dic_summery_action.items():
            ss= dic_action_index[key]
            file.write(f"{ss}: {value}\n")

     with open("./all_episode_length2.txt", "a") as file:
        for key, value in dic_episode_lenth.items():
            file.write(f"{key}: {value}\n")
  
  ss= 1850000
  checkpoint_path="./best_model/dqn_best_model_"+str(ss)+"_steps"
#  #model = DQN.load(checkpoint_path)
  en_for =0
  if(en_for==1):
     model.set_parameters(checkpoint_path)
     model.set_env(env)
     env.startGame(model,ss)
  #plot_actions(dic_test_summery_action,"Test")


  print(dic_episode_lenth)
  print("******************************************Elapsed time: {} hours, {} minutes, {} seconds".format(hours, minutes, seconds))


if __name__ == "__main__":
  game=1 # if you want to play Q learning player
  main(game)
  