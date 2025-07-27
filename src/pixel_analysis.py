import pickle
import os
import torch
import numpy as np

from load_dataset import FrameDataset
from models import Autoencoder
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

def load_dataset(file_path):
    '''Load a previously saved dataset from the specified file path.'''
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)

    return dataset

def load_encoder(file_path):
    '''Load a pre-trained autoencoder from the specified file path.'''
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load(file_path))
    autoencoder.eval()
    
    #Extract and return only the encoder part of the model.
    return autoencoder.encoder

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def generate_image_heatmaps(loader: DataLoader, encoder: Autoencoder, output_dir='heatmaps'):
    to_img = transforms.ToPILImage()
    global_id = 0

    with torch.no_grad():
        for batch_imgs, batch_labels in loader:
            #Get the latent representations of all the images within the batch.
            latent = encoder(batch_imgs) #(batch, channels, height, width)

            #Iterate through each image in the batch.
            for i in range(latent.size(0)):
                features = latent[i] #(channels, height, width)
                label = batch_labels[i].item()

                #Average activation for each pixel across all 128 channels.
                avg_map = features.mean(dim=0)
                avg_np = avg_map.numpy()

                #Maximum activation for each pixel across all 128 channels.
                max_map, _ = features.max(dim=0)
                max_np = max_map.numpy()

                #Images with too few unique values are likely to be anomalies.
                if len(np.unique(avg_np)) < 15 or len(np.unique(max_np)) < 15:
                    print(f"Skipping image {global_id} (label {label}): too few unique values")
                    continue

                #Convert the heatmaps to images
                fname = f'img_{global_id}.png'

                avg_img = to_img(normalize(avg_map))
                avg_img.save(os.path.join(output_dir, f'class_{label}', 'avg', fname))

                max_img = to_img(normalize(max_map))
                max_img.save(os.path.join(output_dir, f'class_{label}', 'max', fname))

                global_id += 1

if __name__ == "__main__":
    encoder = load_encoder("models/autoencoder.pth")

    dataset = load_dataset("datasets/train_classifier.pkl")
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    generate_image_heatmaps(loader, encoder)