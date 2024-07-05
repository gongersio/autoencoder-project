#Import all the pytorch libraries required to define the autoencoder.
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import cv2

#Define a custom dataset class representing all the videos as a collection of frames. 
class FrameDataset(Dataset):
    def __init__(self, dir):
        '''Provide dataset directory.'''
        self.dir = dir

        #All frames should be transformed into pytorch tensors for the autoencoder.
        self.transform = transforms.Compose([
            transforms.Resize((576,1024)),
            transforms.ToTensor()
        ])

        self.frames = []
        self.extract_frames()

    def __len__(self):
        '''Return the total number of frames in the dataset.'''
        return len(self.frames)
    
    def __getitem__(self, idx):
        '''Retrieve a single frame and apply a transformation to it.'''
        frame = self.frames[idx]
        
        #Convert the frame to the correct format before applying the transformation.
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame = self.transform(frame)

        return frame

    def extract_frames(self):
        '''Extract frames from all the videos at specified regular intervals, storing them all in a list.'''
        for video in os.listdir(self.dir):
            video_path = os.path.join(self.dir, video)
            vidcap = cv2.VideoCapture(video_path)
            total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

            #The frame interval is determined assuming the average video is 30 fps.
            for frame_count in range(0, total_frames, 120):
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                success, frame = vidcap.read() #Read the specified frame of the video.

                if success:
                    self.frames.append(frame)
                    print(frame_count)
                
                else:
                    break

            vidcap.release()

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        #Encoder: compresses the input image to a lower-dimensional representation
        self.encoder = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), #Dimensions: 576, 1024, 3 (channels for RGB)
        nn.ReLU(), #Activation Function
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), #Dimensions: 288, 512, 16
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), #Dimensions: 144, 256, 32
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) #Dimensions: 72, 128, 64
        )
        
        #Decoder: reconstructs the image from the lower-dimensional representation
        self.decoder = nn.Sequential(
        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), #Dimensions: 72, 128, 64 
        nn.ReLU(), #Activation Function
        nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1), #Dimensions: 576, 1024, 3
        nn.Sigmoid() #Output pixel values in range [0,1].
        )

    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded