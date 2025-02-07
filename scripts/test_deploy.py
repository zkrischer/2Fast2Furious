#!/usr/bin/env python3
import time
import click
import math
import cv2
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))
from pibot_client import PiBot

from transforms import MyTransforms
from model import Net
from findStopSign import findStopSign

parser = argparse.ArgumentParser(description='PiBot client')
parser.add_argument('--ip', type=str, default='192.168.1.50', help='IP address of PiBot')
args = parser.parse_args()

bot = PiBot(ip=args.ip)

# stop the robot 
bot.setVelocity(0, 0)

#INITIALISE NETWORK HERE
net = Net()
tf = MyTransforms()
from pred2steer import pred2steer

#LOAD NETWORK WEIGHTS HERE
net.load_state_dict(torch.load('steer_net.pth', weights_only=True))

#countdown before beginning
print("Get ready...")
time.sleep(1)
print("3")
time.sleep(1)
print("2")
time.sleep(1)
print("1")
time.sleep(1)
print("GO!")

try:
    angle = 0
    stopped = False
    stopped_count = 0
    max_count = 50
    while True:
        # get an image from the the robot
        im = bot.getImage()
        #TO DO: apply any necessary image transforms
        trans_image = tf(im)
        #TO DO: pass image through network get a prediction
        trans_image = trans_image.unsqueeze(0)

        with torch.inference_mode():
            prediction = net(trans_image)
            print(prediction)
        #TO DO: convert prediction into a meaningful steering angle
        angle = pred2steer(prediction)
        print(angle)

        #TO DO: check for stop signs?
        stop_found = findStopSign(im)
        
        if not stopped and stop_found:
            bot.set_velocity(0.0,0.0)
            stopped = True
            # Need to add the capability to wait
            time.sleep(10)
        elif not stop_found:
            pass
        else:
            stopped_count+=1
            if stopped_count == max_count:
                stopped_count = 0
                stopped = False
        

        Kd = 20 #base wheel speeds, increase to go faster, decrease to go slower
        Ka = 20 #how fast to turn when given an angle
        left  = int(Kd + Ka*angle)
        right = int(Kd - Ka*angle)
            
        bot.setVelocity(left, right)
            
        
except KeyboardInterrupt:    
    bot.setVelocity(0, 0)
