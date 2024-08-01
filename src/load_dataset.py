import os
import cv2
import subprocess
import random
import pickle

#Import all the pytorch libraries required to create the PyTorch dataset.
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image

#Define a custom dataset class that converts all the videos into a collection of representative frames. 
class FrameDataset(Dataset):
    def __init__(self, dir, label):
        '''Provide dataset directory.'''
        self.dir = dir
        self.label = label

        #All frames should be transformed into pytorch tensors of a fixed size for the autoencoder.
        self.transform = transforms.Compose([
            transforms.Resize((256,144)), #Downscale the frames before processing.
            transforms.ToTensor()
        ])

        self.frames = []
        self.extract_frames()

    def __len__(self):
        '''Return the total number of frames in the dataset.'''
        return len(self.frames)
    
    def __getitem__(self, idx):
        '''Retrieve a single frame and apply a transformation to it.'''
        frame = Image.fromarray(self.frames[idx])
        frame = self.transform(frame)

        return frame, self.label

    def extract_frames(self):
        '''For each video, extract all frames where there is a scene change, and then select a random sample of 3 of those frames.'''
        for video in os.listdir(self.dir):
            if video == ".frames": continue #Must exclude the 'frames' folder.
            video_path = os.path.join(self.dir, video)

            #Create a temporary folder to store all the extracted frames.
            output_path = os.path.join(self.dir, ".frames")
            os.makedirs(output_path, exist_ok=True)
            
            #The ffmpeg command to extract a frame at each scene change in the video.
            #A scene change occurs when two consecutive frames exceed the difference threshold (30%).
            #In this case, the first scene is also counted as a scene change.
            cmd = (
                f"ffmpeg -loglevel error -i \"{video_path}\" "
                f"-vf \"select='eq(n\\,0)+gt(scene\\,0.3)'\" " 
                f"-fps_mode vfr -frame_pts true \"{output_path}/img_%04d.jpg\""
            )

            subprocess.run(cmd, shell=True, check=True)

            #Choose a random sample of 5 from all the extracted scene change frames.  
            extracted_frames = [os.path.join(output_path, f) for f in os.listdir(output_path)]
            frame_sample = random.sample(extracted_frames, min(len(extracted_frames), 5))

            for frame_path in frame_sample:
                frame = cv2.imread(frame_path)
                self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #Convert the frame to the correct format.

            #Clean up the temporary files.
            for f in extracted_frames:
                os.remove(f)

        os.rmdir(output_path)

if __name__ == "__main__":
    #The model should be trained on both types of video - tags are used to distinguish them.
    nat_hist_dataset = FrameDataset("../natural_history_museum/training_videos", 0)
    frameless_dataset = FrameDataset("../frameless/training_videos", 1)
    combined_dataset = ConcatDataset([frameless_dataset, nat_hist_dataset])

    #Save the combined dataset to avoid repeating the frame extraction process.
    with open('combined_dataset.pkl', 'wb') as f:
        pickle.dump(combined_dataset, f)