import random
import torch
import pickle

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class TripletDataset(Dataset):
    '''Transforms a standard FrameDataset and prepares it to produce triplets (anchor, positive, negative) for model training. '''
    def __init__(self, dataset):
        self.dataset = dataset
        self.labels = [label for _, label in self.dataset]
        self.labels_to_indices = self.map_labels_to_indices()

    def map_labels_to_indices(self):
        '''Create a mapping between each class label and all its corresponding indices in the dataset.'''
        labels_to_indices = {}

        for idx, label in enumerate(self.labels):
            if label not in labels_to_indices:
                labels_to_indices[label] = []

            labels_to_indices[label].append(idx)

        return labels_to_indices

    def __len__(self):
        '''Return the total number of frames in the dataset.'''
        return len(self.dataset)

    def __getitem__(self, idx):
        '''Generates and returns one triplet.'''
        #Get an anchor image and its label.
        anchor_img, anchor_label = self.dataset[idx]

        #Get a positive image (different image from the same class).
        positive_list = self.labels_to_indices[anchor_label]
        positive_idx = idx

        #Ensure the positive image is different from the anchor image.
        while positive_idx == idx:
            positive_idx = random.choice(positive_list)
            
        positive_img, _ = self.dataset[positive_idx]

        #Get a negative image (image from the other class).
        negative_label = 1 - anchor_label
        negative_list = self.labels_to_indices[negative_label]
        negative_idx = random.choice(negative_list)
        negative_img, _ = self.dataset[negative_idx]

        return anchor_img, positive_img, negative_img, anchor_label
