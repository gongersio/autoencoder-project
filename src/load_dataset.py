import os
import cv2
import subprocess
import re
import random
import pickle
import tempfile
import numpy as np
from PIL import Image

#Import the necessary pytorch libraries.
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms

def blurriness(image_path):
    '''Determine if the given image is blurry by calculating the Laplacian variance.'''
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    return laplacian_var

class FrameDataset(Dataset):
    def __init__(self, dir, label, augment=False, save_frames=None, start_id=0, return_id=False):
        '''Converts all videos in the given directory into a representative frame dataset.'''
        self.dir = dir
        self.label = label #Labels are used to separate the videos into distinct classes.
        self.save_frames = save_frames
        self.start_id = start_id
        self.return_id = return_id #Return each video ID when returned for final model predictions.

        if self.save_frames:
            os.makedirs(self.save_frames, exist_ok=True)

            for f in os.listdir(self.save_frames):
                os.remove(os.path.join(self.save_frames, f)) #Remove existing saved frames to avoid conflicts. 

        #For training data, the transformation should include data augmentations to make the model more robust.
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((1024, 576)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                transforms.ToTensor()
            ])

        #For validation/test data, the transformation should only resize the frames.
        else:
            self.transform = transforms.Compose([
                transforms.Resize((1024, 576)),
                transforms.ToTensor()
            ])

        self.frames = []
        self.videos_processed = self.extract_frames()

    def __len__(self):
        '''Return the total number of frames in the dataset.'''
        return len(self.frames)
    
    def __getitem__(self, idx):
        '''Retrieve a single frame and apply a transformation to it.'''
        frame_data, video_id = self.frames[idx]
        frame = Image.fromarray(frame_data)
        frame = self.transform(frame)

        if self.return_id:
            return frame, self.label, video_id

        else:
            return frame, self.label

    def extract_frames(self, sample_size=7, fps=3):
        '''For each video, extract scene change frames, and then select a uniformly spaced sample of up to 7 frames.'''
        video_id = self.start_id

        for video in os.listdir(self.dir):
            video_path = os.path.join(self.dir, video)
            if not os.path.isfile(video_path): continue

            #Create a temporary folder to store all the extracted frames.
            with tempfile.TemporaryDirectory() as temp_dir:
            
                #ffmpeg command to get the timestamp of each scene change in the video (difference threshold > 30%).
                #The first frame is counted as a scene change.
                cmd_times = (
                        f"ffmpeg -i \"{video_path}\" "
                        f"-vf \"select='eq(n\\,0)+gt(scene\\,0.3)',showinfo\" -f null -"
                    )

                result = subprocess.run(cmd_times, shell=True, capture_output=True, text=True)
                timestamps = re.findall(r"pts_time:([\d\.]+)", result.stderr)

                final_frames = []
                for idx, ts in enumerate(timestamps):
                    start_time = float(ts)

                    #Create a 1 second window around each scene change, excluding the first frame.
                    if start_time > 0:
                        start_time = max(0, start_time - 1)

                    #ffmpeg command to extract frames within the scene change window.
                    cmd_frames = (
                        f"ffmpeg -loglevel error -hwaccel cuda -i \"{video_path}\" -ss {start_time} -t 1 "
                        f"-vf \"fps={fps}\" \"{temp_dir}/frame_{idx}_img_%04d.png\""
                    )

                    subprocess.run(cmd_frames, shell=True, check=True)

                    #Get all the frames extracted in this scene change window.
                    window_frames = [os.path.join(temp_dir, f) for f in sorted(os.listdir(temp_dir)) if f.startswith(f"frame_{idx}_")]

                    #If only one frame was extracted, use it. Otherwise, find the least blurry one from the window.
                    if len(window_frames) == 1:
                        frame_choice = window_frames[0]

                    else:
                        variances = [blurriness(f) for f in window_frames]
                        best_index = np.argmax(variances)
                        frame_choice = window_frames[best_index]

                    final_frames.append(frame_choice)

                if not final_frames:
                    print(f"No suitable frames detected in video {video}.")
                    video_id += 1
                    continue
                
                #If there are more than 7 (sample_size) frames, select a uniform sample.
                if len(final_frames) > sample_size:
                    indices = np.linspace(0, len(final_frames) - 1, sample_size, dtype=int)
                    frame_sample = [final_frames[i] for i in indices]

                else:
                    frame_sample = final_frames

                #Ensure the number of frames is odd for majority voting.
                #If the count is even, randomly remove a frame from the sample.
                if len(frame_sample) % 2 == 0:
                    frame_to_remove = random.choice(frame_sample)
                    frame_sample.remove(frame_to_remove)

                frame_id = 0
                for frame_path in frame_sample:
                    frame = cv2.imread(frame_path)

                    if self.save_frames:
                        save_name = f"vid{video_id}_frame{frame_id}.png"
                        save_path = os.path.join(self.save_frames, save_name)
                        cv2.imwrite(save_path, frame)
                    
                    self.frames.append((cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), video_id))
                    frame_id += 1

            print(f"Processed video {video_id}")
            video_id += 1
        
        return video_id - self.start_id

if __name__ == "__main__":
    #Create a frame dataset that will be used to train the model.
    nat_hist_train = FrameDataset("../natural_history_museum/training", 0, augment=True)
    frameless_train = FrameDataset("../frameless/training", 1, augment=True)
    train_dataset = ConcatDataset([nat_hist_train, frameless_train]),

    #Create a frame dataset without augmentations that will be used to analyse image trends.
    nat_hist_analyse = FrameDataset("../natural_history_museum/training", 0)
    frameless_analyse = FrameDataset("../frameless/training", 1)
    analyse_dataset = ConcatDataset([nat_hist_analyse, frameless_analyse])

    #Create a frame dataset that will be used to validate the model.
    nat_hist_eval = FrameDataset("../natural_history_museum/validation", 0)
    frameless_eval = FrameDataset("../frameless/validation", 1)
    eval_dataset = ConcatDataset([nat_hist_eval, frameless_eval])

    #Create a frame dataset that will be used to test the model.
    num_videos = 0

    nat_hist_test = FrameDataset("../natural_history_museum/testing", 0, save_frames="../natural_history_museum/test_frames", return_id=True)
    num_videos += nat_hist_test.videos_processed
    frameless_test = FrameDataset("../frameless/testing", 1, save_frames="../frameless/test_frames", start_id=num_videos, return_id=True)

    test_dataset = ConcatDataset([nat_hist_test, frameless_test])
    
    with open('datasets/train_model.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)

    with open('datasets/analyse_model.pkl', 'wb') as f:
        pickle.dump(analyse_dataset, f)

    with open('datasets/eval_model.pkl', 'wb') as f:
        pickle.dump(eval_dataset, f)

    with open('datasets/test_model.pkl', 'wb') as f:
        pickle.dump(test_dataset, f)