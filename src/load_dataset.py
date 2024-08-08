import os
import cv2
import subprocess
import random
import pickle
from PIL import Image

#Import the necessary pytorch libraries.
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms

class FrameDataset(Dataset):
    def __init__(self, dir, label=0):
        '''Converts all videos in the given directory into a representative frame dataset.'''
        self.dir = dir
        self.label = label #Labels are used to separate the videos into distinct classes.

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
        '''For each video, extract all frames where there is a scene change, and then select a random sample including 5 of those frames.'''
        for video in os.listdir(self.dir):
            if video == ".frames": continue #Must exclude the temporary '.frames' folder.
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

            #Choose a random sample of 5 out of all the extracted scene change frames.  
            extracted_frames = [os.path.join(output_path, f) for f in os.listdir(output_path)]
            frame_sample = random.sample(extracted_frames, min(len(extracted_frames), 5))

            #Convert each frame to the correct format before it is added to the dataset.
            for frame_path in frame_sample:
                frame = cv2.imread(frame_path)
                self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 

            #Clean up all the temporary files.
            for f in extracted_frames:
                os.remove(f)

        os.rmdir(output_path)

if __name__ == "__main__":
    #Create a frame dataset that will be used to train the autoencoder.
    nat_hist_train = FrameDataset("../natural_history_museum/autoencoder_training")
    frameless_train = FrameDataset("../frameless/autoencoder_training")
    train_autoencoder = ConcatDataset([nat_hist_train, frameless_train])

    #Create a frame dataset that will be used to validate the autoencoder.
    nat_hist_eval = FrameDataset("../natural_history_museum/autoencoder_validation")
    frameless_eval = FrameDataset("../frameless/autoencoder_validation")
    eval_autoencoder = ConcatDataset([nat_hist_eval, frameless_eval])

    #Create a frame dataset (with labels) that will be used to train the classifer.
    nat_hist_train = FrameDataset("../natural_history_museum/classifier_training", 0)
    frameless_train = FrameDataset("../frameless/classifier_training", 1)
    train_classifier = ConcatDataset([nat_hist_train, frameless_train])

    #Create a frame dataset (with labels) that will be used to validate the classifer.
    nat_hist_train = FrameDataset("../natural_history_museum/classifier_validation", 0)
    frameless_train = FrameDataset("../frameless/classifier_validation", 1)
    eval_classifier = ConcatDataset([nat_hist_train, frameless_train])

    #Save all the datasets to avoid repeating the frame extraction process.
    with open('train_autoencoder.pkl', 'wb') as f:
        pickle.dump(train_autoencoder, f)
    
    with open('eval_autoencoder.pkl', 'wb') as f:
        pickle.dump(eval_autoencoder, f)

    with open('train_classifier.pkl', 'wb') as f:
        pickle.dump(train_classifier, f)

    with open('eval_classifier.pkl', 'wb') as f:
        pickle.dump(eval_classifier, f)