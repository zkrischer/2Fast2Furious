import numpy as np
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
from glob import glob
from os import path

from transforms import MyTransforms

class SteerDataSet(Dataset):
    
    def __init__(self,root_folder,img_ext = ".jpg"):
        self.root_folder = root_folder
        self.transform = MyTransforms()     
        self.img_ext = img_ext        
        self.filenames = glob(path.join(self.root_folder,"*" + self.img_ext))            
        self.class_labels = ['sharp left',
                            'left',
                            'straight',
                            'right',
                            'sharp right']
        
    def __len__(self):        
        return len(self.filenames)
    
    def __getitem__(self,idx):
        f = self.filenames[idx]        
        img = cv2.imread(f)[120:, :, :] 
        
        steering = path.split(f)[-1].split(self.img_ext)[0][6:]
        steering = float(steering)

        img, steering = self.transform(img, steering)     

        if steering <= -0.5:
            steering_cls = 0
        elif steering < 0:
            steering_cls = 1
        elif steering == 0:
            steering_cls = 2
        elif steering < 0.5:
            steering_cls = 3
        else:
            steering_cls = 4 
                      
        return img, steering_cls
