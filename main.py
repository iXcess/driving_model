#!/usr/bin/env python3
from common.transformations.camera import transform_img, eon_intrinsics
from common.transformations.model import medmodel_intrinsics
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

import cv2 
from tensorflow.keras.models import load_model
from tools.lib.parser import parser
import cv2
import sys
camerafile = sys.argv[1]
supercombo = load_model('supercombo.keras')

MAX_DISTANCE = 140.
LANE_OFFSET = 1.8
MAX_REL_V = 10.

LEAD_X_SCALE = 10
LEAD_Y_SCALE = 10

cap = cv2.VideoCapture(camerafile)

imgs = []

for i in tqdm(range(1000)):
  ret, frame = cap.read()
  img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
  imgs.append(img_yuv.reshape((874*3//2, 1164)))
 

def frames_to_tensor(frames):                                                                                               
  H = (frames.shape[1]*2)//3                                                                                                
  W = frames.shape[2]                                                                                                       
  in_img1 = np.zeros((frames.shape[0], 6, H//2, W//2), dtype=np.uint8)                                                      
                                                                                                                            
  in_img1[:, 0] = frames[:, 0:H:2, 0::2]                                                                                    
  in_img1[:, 1] = frames[:, 1:H:2, 0::2]                                                                                    
  in_img1[:, 2] = frames[:, 0:H:2, 1::2]                                                                                    
  in_img1[:, 3] = frames[:, 1:H:2, 1::2]                                                                                    
  in_img1[:, 4] = frames[:, H:H+H//4].reshape((-1, H//2,W//2))                                                              
  in_img1[:, 5] = frames[:, H+H//4:H+H//2].reshape((-1, H//2,W//2))
  return in_img1

imgs_med_model = np.zeros((len(imgs), 384, 512), dtype=np.uint8)
for i, img in tqdm(enumerate(imgs)):
  imgs_med_model[i] = transform_img(img, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                    output_size=(512,256))
frame_tensors = frames_to_tensor(np.array(imgs_med_model)).astype(np.float32)/128.0 - 1.0


state = np.zeros((1,512))
desire = np.zeros((1,8))

cap = cv2.VideoCapture(camerafile)

for i in tqdm(range(len(frame_tensors) - 1)):
  inputs = [np.vstack(frame_tensors[i:i+2])[None], desire, state]
  outs = supercombo.predict(inputs)
  parsed = parser(outs)
  # Important to refeed the state
  state = outs[-1]
  pose = outs[-2]
  ret, frame = cap.read()
  frame = cv2.resize(frame, (640, 420))
  # Show raw camera image
  cv2.imshow("modeld", frame)
  # Clean plot for next frame
  plt.clf()
  plt.title("lanes and path")
  # lll = left lane line
  plt.plot(parsed["lll"][0], range(0,192), "b-", linewidth=1)
  # rll = right lane line
  plt.plot(parsed["rll"][0], range(0, 192), "r-", linewidth=1)
  # path = path cool isn't it ?
  plt.plot(parsed["path"][0], range(0, 192), "g-", linewidth=1)
  #print(np.array(pose[0,:3]).shape)
  #plt.scatter(pose[0,:3], range(3), c="y")
  
  # Needed to invert axis because standart left lane is positive and right lane is negative, so we flip the x axis
  plt.gca().invert_xaxis()
  plt.pause(0.001)
  if cv2.waitKey(10) & 0xFF == ord('q'):
        break

plt.show()
